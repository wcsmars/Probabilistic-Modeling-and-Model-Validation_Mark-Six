"""Behavioral checks for leakage, mixture arithmetic and synthetic controls."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from marksix.cli import run
from marksix.data import load_draws, write_synthetic
from marksix.evaluation import MIXTURE_PRIOR, block_uncertainty, holm, summarize, walk_forward


def synthetic(n=90, seed=17, planted=False):
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 49))
    for row in y:
        indices = np.r_[0, rng.choice(np.arange(1, 49), 5, replace=False)] if planted else rng.choice(49, 6, replace=False)
        row[indices] = 1
    return y


class EvaluationTests(unittest.TestCase):
    def test_target_and_future_do_not_change_forecasts(self):
        y = synthetic()
        altered = y.copy()
        altered[65:] = synthetic(25, seed=131)
        original = walk_forward(y, warmup=50)
        changed = walk_forward(altered, warmup=50)
        # Forecast for target 65 must also be independent of its own outcome.
        np.testing.assert_allclose(original['marginals'][:16], changed['marginals'][:16], atol=0, rtol=0)
        np.testing.assert_allclose(original['mixture_weights'][:16], changed['mixture_weights'][:16], atol=0, rtol=0)
        np.testing.assert_allclose(original['log_probabilities'][:15], changed['log_probabilities'][:15], atol=0, rtol=0)

    def test_arithmetic_mixture_scores_and_weight_replay(self):
        result = walk_forward(synthetic(55), warmup=30)
        lp, weights = result['log_probabilities'], result['mixture_weights']
        np.testing.assert_allclose(lp[:, 3], logsumexp(np.log(weights) + lp[:, :3], axis=1), atol=1e-12)
        expected = MIXTURE_PRIOR.copy()
        for row, saved in zip(lp, weights):
            np.testing.assert_allclose(saved, expected, atol=1e-14)
            posterior = expected * np.exp(.25 * (row[:3] - row[:3].max()))
            posterior /= posterior.sum()
            expected = .99 * posterior + .01 * MIXTURE_PRIOR

    def test_reset_removes_old_regime_from_new_forecasts(self):
        y = synthetic(80)
        changed = y.copy()
        changed[:60] = synthetic(60, seed=123, planted=True)
        a = walk_forward(y, warmup=50, reset_index=60)
        b = walk_forward(changed, warmup=50, reset_index=60)
        np.testing.assert_allclose(a['marginals'][10:], b['marginals'][10:], atol=0, rtol=0)
        np.testing.assert_allclose(a['mixture_weights'][10], MIXTURE_PRIOR)
        np.testing.assert_allclose(a['marginals'][10], np.full((4, 49), 6/49), atol=1e-12)

    def test_strong_synthetic_signal_is_detectable(self):
        result = walk_forward(synthetic(140, planted=True), warmup=60)
        gain = result['log_probabilities'][:, 1:] - result['log_probabilities'][:, :1]
        self.assertTrue(np.all(gain.sum(axis=0) > 20))

    def test_uniform_control_and_probability_constraints(self):
        y = synthetic(75)
        result = walk_forward(y, warmup=60)
        np.testing.assert_allclose(result['marginals'].sum(axis=-1), 6, atol=1e-11)
        self.assertTrue(np.all((result['marginals'] >= 0) & (result['marginals'] <= 1)))
        summary = summarize(result, y)
        self.assertAlmostEqual(summary[0]['total_log_gain'], 0., places=10)
        self.assertAlmostEqual(summary[0]['mean_brier_gain'], 0., places=10)
        self.assertEqual(summary[0]['holm_p'], 1.)

    def test_short_samples_do_not_emit_confident_inference(self):
        with self.assertRaises(ValueError):
            block_uncertainty(np.ones((8, 3)))
        y = synthetic(68, planted=True)
        summary = summarize(walk_forward(y, warmup=60), y)
        self.assertIsNone(summary[1]['mean_95pct_block_ci'])
        self.assertIsNone(summary[1]['holm_p'])
        with self.assertRaises(ValueError):
            walk_forward(synthetic(65).astype(complex) + 1j, warmup=60)

    def test_holm_reference_and_invalid_arguments(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        with self.assertRaises(ValueError):
            walk_forward(synthetic(10), warmup=10)
        with self.assertRaises(ValueError):
            walk_forward(synthetic(10), warmup=5, reset_index=-1)
        with self.assertRaises(ValueError):
            holm([float('nan')])


class DataAndRunTests(unittest.TestCase):
    def test_synthetic_regeneration_and_strict_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a.csv', Path(directory)/'b.csv'
            write_synthetic(a, draws=25)
            write_synthetic(b, draws=25)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(load_draws(a).outcomes.shape, (25, 49))
            with self.assertRaises(FileExistsError):
                write_synthetic(a, draws=25)

    def test_rejects_disordered_duplicate_and_invalid_draws(self):
        header = ['date', 'draw_id', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra']
        good = ['2000-01-01', 'A', 1, 2, 3, 4, 5, 6, 7]
        cases = [
            [good, ['1999-12-31', 'B', 1, 2, 3, 4, 5, 6, 7]],
            [good, ['2000-01-02', 'A', 1, 2, 3, 4, 5, 6, 7]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, 5, 7]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, 50, 7]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, 6, 6]],
            [['2000-01-01', 'A', 1, 2, 3, 4, 5, '6.5', 7]],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.csv'
            for rows in cases:
                with self.subTest(rows=rows):
                    with path.open('w', newline='') as handle:
                        writer = csv.writer(handle); writer.writerow(header); writer.writerows(rows)
                    with self.assertRaises(ValueError):
                        load_draws(path)

    def test_evaluation_preserves_inputs_and_prior_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/'input.csv', root/'output'
            write_synthetic(source, draws=75)
            before = source.read_bytes()
            summary = run(source, output)
            self.assertEqual(summary['dataset_kind'], 'user_supplied_unverified')
            self.assertEqual(source.read_bytes(), before)
            saved = (output/'summary.json').read_bytes()
            with self.assertRaises(ValueError):
                run(source, output)
            self.assertEqual(saved, (output/'summary.json').read_bytes())
            self.assertFalse((output/'synthetic_draws.csv').exists())
            self.assertNotIn(str(root), (output/'provenance.json').read_text())

    def test_invalid_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic(root/'input.csv', draws=10)
            with self.assertRaises(ValueError):
                run(root/'input.csv', root/'output', warmup=60)
            self.assertFalse((root/'output').exists())

    def test_evaluation_uses_one_snapshot_when_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, replacement, output = root/'input.csv', root/'replacement.csv', root/'output'
            write_synthetic(source, draws=65)
            write_synthetic(replacement, draws=80, seed=99)
            original = source.read_bytes()

            def replace_during_evaluation(*args, **kwargs):
                replacement.replace(source)
                return walk_forward(*args, **kwargs)

            with patch('marksix.cli.walk_forward', side_effect=replace_during_evaluation):
                summary = run(source, output, dataset_kind='synthetic_demo')

            self.assertNotEqual(source.read_bytes(), original)
            self.assertEqual(summary['input_draws'], 65)
            self.assertEqual(summary['evaluation_draws'], 5)
            self.assertEqual((output/'synthetic_draws.csv').read_bytes(), original)
            provenance = json.loads((output/'provenance.json').read_text())
            self.assertEqual(provenance['input_sha256'], hashlib.sha256(original).hexdigest())
