"""Fit a conservative signed box from already collected pulse CSVs. Never moves hardware."""

import argparse
import csv
from dataclasses import asdict, replace
import math
from pathlib import Path

import yaml

from .settings import load_settings

DIRECTIONS = ("x+", "x-", "y+", "y-", "z+", "z-")


def fit_box(rows, *, tolerance=0.05, min_repeats=3):
    if not 0 < tolerance < 1 or min_repeats < 1:
        raise ValueError("invalid fit parameters")
    groups = {key: {} for key in DIRECTIONS}
    for row in rows:
        axis = row["axis"]
        level, nominal, actual = (float(row[k]) for k in ("command_m_per_step", "nominal_m", "measured_m"))
        if axis not in groups or not all(math.isfinite(x) for x in (level, nominal, actual)) or level <= 0 or nominal == 0:
            raise ValueError(f"invalid pulse row: {row}")
        if (nominal > 0) != axis.endswith("+"):
            raise ValueError("nominal displacement sign must match axis")
        groups[axis].setdefault(level, []).append(abs(actual / nominal - 1))
    limits = {}
    for axis, levels in groups.items():
        bound = None
        for level, errors in sorted(levels.items()):
            # Once a level fails, do not accept a disconnected higher level.
            if len(errors) < min_repeats or max(errors) > tolerance + 1e-12:
                break
            bound = level
        if bound is None:
            raise ValueError(f"no contiguous passing levels for {axis}; need >= {min_repeats} repeats per level")
        limits[axis] = bound
    return {"units": "metres_per_action_step", "positive": [limits[a + "+"] for a in "xyz"],
            "negative": [limits[a + "-"] for a in "xyz"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate-cap", type=float, default=1.0, help="Select on development trials, not pulse data")
    parser.add_argument("--min-repeats", type=int, default=3)
    args = parser.parse_args()
    with args.csv.open() as stream:
        box = fit_box(csv.DictReader(stream), min_repeats=args.min_repeats)
    box["source"] = f"{args.csv.resolve()}; all repeats within 5%; >= {args.min_repeats} per level"
    settings = replace(load_settings(args.base_config), actuator=box, rate_cap=args.rate_cap,
                       notes="Box fitted from pulse CSV; does not validate contact tasks or select the speed cap.")
    with args.output.open("x") as stream:
        yaml.safe_dump(asdict(settings), stream, sort_keys=False)
    print(args.output)


if __name__ == "__main__":
    main()
