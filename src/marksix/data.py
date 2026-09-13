"""Strict draw ingestion and an entirely synthetic demonstration dataset."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
import io
from pathlib import Path
import random

import numpy as np


@dataclass(frozen=True)
class DrawData:
    dates: tuple[str, ...]
    draw_ids: tuple[str, ...]
    outcomes: np.ndarray


def load_draws(path: str | Path) -> DrawData:
    """Load a CSV file as a single snapshot."""
    return load_draws_bytes(Path(path).read_bytes())


def load_draws_bytes(content: bytes) -> DrawData:
    """Read chronological, unique dated rows; never silently sort or deduplicate."""
    dates, ids, outcomes = [], [], []
    with io.StringIO(content.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "draw_id", *(f"n{i}" for i in range(1, 7))}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("CSV requires date, draw_id, and n1 through n6")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV header names must be unique")
        for line, row in enumerate(reader, start=2):
            try:
                if None in row or any(row[key] is None for key in required):
                    raise ValueError("Malformed CSV row")
                day = date.fromisoformat(row["date"])
                if day.isoformat() != row["date"]:
                    raise ValueError("date must use YYYY-MM-DD")
                draw_id = row["draw_id"].strip()
                if not draw_id or draw_id in ids:
                    raise ValueError("draw_id must be nonempty and unique")
                if dates and day.isoformat() <= dates[-1]:
                    raise ValueError("Dates must be unique and strictly increasing")
                balls = [int(row[f"n{i}"]) for i in range(1, 7)]
                if len(set(balls)) != 6 or any(b < 1 or b > 49 for b in balls):
                    raise ValueError("Mains must be six distinct integers from 1 to 49")
                if row.get("extra", "").strip():
                    extra = int(row["extra"])
                    if extra not in range(1, 50) or extra in balls:
                        raise ValueError("extra must be a distinct integer from 1 to 49")
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid draw at CSV line {line}: {error}") from error
            indicator = np.zeros(49, dtype=float)
            indicator[np.asarray(balls) - 1] = 1
            dates.append(day.isoformat())
            ids.append(draw_id)
            outcomes.append(indicator)
    if not outcomes:
        raise ValueError("CSV has no draws")
    return DrawData(tuple(dates), tuple(ids), np.stack(outcomes))


def write_synthetic(path: str | Path, *, draws: int = 240, seed: int = 20260914) -> None:
    """Generate independent fair draws; dates are artificial daily indices."""
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 2:
        raise ValueError("draws must be an integer of at least two")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["date", "draw_id", *(f"n{i}" for i in range(1, 7))])
        for index in range(draws):
            balls = sorted(rng.sample(range(1, 50), 6))
            writer.writerow([(date(2000, 1, 1) + timedelta(days=index)).isoformat(),
                             f"SYN-{index + 1:04d}", *balls])
