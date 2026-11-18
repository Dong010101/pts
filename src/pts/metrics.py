"""Pooled throughput from scored trial CSVs; include successful and failed attempts."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


def summarize(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["method"], row["trial_id"])
        if key in seen:
            raise ValueError(f"duplicate trial: {key}")
        seen.add(key)
        success, duration, steps = int(row["success"]), float(row["duration_s"]), float(row["control_steps"])
        if success not in (0, 1) or not all(math.isfinite(x) and x > 0 for x in (duration, steps)):
            raise ValueError("each trial needs success 0/1 and positive finite duration/steps")
        groups[row["method"]].append((success, duration, steps))
    result = {}
    for method, trials in sorted(groups.items()):
        successes = sum(t[0] for t in trials)
        seconds, steps = sum(t[1] for t in trials), sum(t[2] for t in trials)
        result[method] = {
            "attempts": len(trials), "successes": successes, "success_rate": successes / len(trials),
            "mean_attempt_s": seconds / len(trials), "total_duration_s": seconds,
            "successes_per_hour": 3600 * successes / seconds,
            "successes_per_1000_steps": 1000 * successes / steps,
            "mean_success_s": sum(t[1] for t in trials if t[0]) / successes if successes else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="method,trial_id,success,duration_s,control_steps")
    args = parser.parse_args()
    with args.csv.open() as stream:
        print(json.dumps(summarize(csv.DictReader(stream)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
