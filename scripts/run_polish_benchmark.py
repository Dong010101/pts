"""Freeze/run the PTS polish comparison, referencing the existing checkpoint only."""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checkpoint_inventory(cfg):
    """Reuse the archive's full-file SHA ledger; recheck stat/permissions, not copy."""
    checkpoint = Path(cfg["checkpoint"])
    manifest = Path(cfg["checkpoint_archive_manifest"])
    archive = json.loads(manifest.read_text())
    if Path(archive["destination"]).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint does not match archived destination")
    files = {}
    for record in archive["files"]:
        path = checkpoint / record["path"]
        stat = path.stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["source_mtime_ns"] or stat.st_mode & 0o222:
            raise ValueError(f"Archived file size/mtime/read-only check failed: {path}")
        files[record["path"]] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                  "mode": stat.st_mode, "inode": stat.st_ino, "device": stat.st_dev}
    return {"checkpoint": str(checkpoint.resolve()), "archive_manifest_sha256": digest(manifest), "files": files,
            "total_bytes": sum(v["bytes"] for v in files.values()), "new_checkpoint_bytes": 0,
            "verification": "Reused prior full SHA256 archive ledger; checked sizes/mtimes/read-only flags/inodes before and after. No new full-weight rehash."}


def prepare(source, out, analysis_python):
    cfg = json.loads((source / "settings.json").read_text())
    for name, expected in json.loads((source / "code_sha256.json").read_text()).items():
        if digest(source / "code" / name) != expected:
            raise ValueError(f"FreeSpeed frozen source changed: {name}")
    if digest(source / "fixed_inputs.npz") != cfg["fixed_inputs_sha256"]:
        raise ValueError("FreeSpeed frozen input checksum mismatch")
    before = checkpoint_inventory(cfg)
    out.mkdir(parents=True, exist_ok=False)
    (out / "code").mkdir()
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project, out / "code/PTS", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".venv", ".pytest_cache", "artifacts"))
    # All large/shared resources stay at their original paths; symlinks make the
    # dependency explicit. No checkpoint directory is created under this run.
    for target, link in ((source / "code/openpi", out / "code/openpi"),
                         (source / "cache", out / "cache"),
                         (source / "fixed_inputs.npz", out / "fixed_inputs.npz"),
                         (source / "fixed_observations.jpg", out / "fixed_observations.jpg")):
        link.symlink_to(target, target_is_directory=target.is_dir())
    cfg.update(
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), source_freespeed_run=str(source),
        pts_snapshot=str(out / "code/PTS"), openpi_snapshot=str(out / "code/openpi"),
        fixed_inputs=str(out / "fixed_inputs.npz"), shared_cache=str(out / "cache"),
        shared_benchmark_helpers=str(source / "code/FreeSpeed/scripts/benchmark_polish.py"),
        analysis_python=analysis_python,
        conditions=["baseline_10step", "pts_1x", "pts_1p5x", "pts_2x"],
        pts_core={"replan_steps": 8, "rate_grid": 0.25, "latency_steps": 2,
                  "gripper_min_steps": 2, "gripper_max_steps": 9,
                  "gripper_speed_threshold": 0.02, "gripper_command_threshold": 0.02},
        offline_actuator_box={"positive": [0.035] * 3, "negative": [0.035] * 3,
                              "max_translation_norm": 0.035, "max_rotation_travel": 0.060},
        offline_box_source="Diagnostic limits (35 mm / 60 mrad) matching FreeSpeed offline step checks; NOT a calibrated Flexiv actuator box.",
        prefix_rule="H=50; stop at first consumed progress >= s=8. Rate 1.5 may end at progress 9.",
        decoder_design="Baseline: same frozen Gaussian, 10 steps. PTS: zero latent, 1 step, same model weights/postprocessing.",
        comparison_design="Four interleaved conditions in one loaded model; separate single-condition processes for memory.",
        checkpoint_copy_policy="Reference the existing immutable archive. No weight copy, export or checkpoint symlink directory.",
    )
    cfg.pop("rates", None)
    cfg.pop("num_steps", None)
    write(out / "settings.json", cfg)
    write(out / "checkpoint_before.json", before)
    write(out / "code_sha256.json", {str(p.relative_to(out / "code/PTS")): digest(p)
          for p in sorted((out / "code/PTS").rglob("*")) if p.is_file()})
    write(out / "external_references.json", {"source_run": str(source),
          "source_code_manifest_sha256": digest(source / "code_sha256.json"),
          "source_settings_sha256": digest(source / "settings.json"),
          "fixed_inputs_sha256": digest(out / "fixed_inputs.npz"),
          "helper_sha256": digest(cfg["shared_benchmark_helpers"]),
          "tokenizer_sha256": digest(out / "cache/big_vision/paligemma_tokenizer.model")})
    write(out / "environment.json", {"python": sys.version, "executable": sys.executable,
          "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                  if d.metadata.get("Name")))})
    command = shlex.join([sys.executable, str(out / "code/PTS/scripts/run_polish_benchmark.py"),
                         "--source-run", str(source), "--analysis-python", analysis_python])
    (out / "reproduce.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + command + ' --output "${1:?Specify a NEW result directory}"\n')
    return cfg


def execute(run, cfg):
    for condition in ("paired", *cfg["conditions"]):
        print(f"Running {condition}: {run / (condition + '.log')}", flush=True)
        with (run / f"{condition}.log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(run / "code/PTS/scripts/benchmark_polish.py"),
                            "--run", str(run), "--condition", condition], stdout=log, stderr=subprocess.STDOUT, check=True)
    after = checkpoint_inventory(cfg)
    if after != json.loads((run / "checkpoint_before.json").read_text()):
        raise AssertionError("Checkpoint changed during the benchmark")
    write(run / "checkpoint_after.json", after)
    subprocess.run([cfg["analysis_python"], str(run / "code/PTS/scripts/analyze_polish_benchmark.py"),
                    "--run", str(run)], check=True)
    print(f"Completed: {run / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path, help="Execute an already prepared run")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analysis-python", default="/usr/bin/python3")
    args = parser.parse_args()
    if args.run:
        run = args.run.resolve()
        cfg = json.loads((run / "settings.json").read_text())
    else:
        if not args.source_run or not args.output:
            parser.error("--source-run and --output are required when preparing")
        run = args.output.resolve()
        cfg = prepare(args.source_run.resolve(), run, args.analysis_python)
    print(f"Prepared {run}; checkpoint reused: {cfg['checkpoint']}", flush=True)
    if not args.prepare_only:
        execute(run, cfg)


if __name__ == "__main__":
    main()
