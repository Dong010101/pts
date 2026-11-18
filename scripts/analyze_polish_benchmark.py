"""Analyze PTS offline measurements; no inference, task-success or hardware claims."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def pose_at(state, actions, progress):
    poses = np.vstack((state, actions))
    grid = np.arange(len(poses))
    xyz = np.stack([np.interp(progress, grid, poses[:, j]) for j in range(3)], axis=-1)
    rotation = Slerp(grid, Rotation.from_rotvec(poses[:, 3:6]))(progress).as_rotvec()
    gripper = poses[np.ceil(progress - 1e-10).astype(int), 6]
    return np.column_stack((xyz, rotation, gripper))


def errors(actual, expected):
    pos = np.linalg.norm(actual[:, :3] - expected[:, :3], axis=1) * 1000
    rot = np.rad2deg((Rotation.from_rotvec(actual[:, 3:6]) * Rotation.from_rotvec(expected[:, 3:6]).inv()).magnitude())
    return {"position_mean_mm": float(pos.mean()), "position_max_mm": float(pos.max()),
            "rotation_mean_deg": float(rot.mean()), "rotation_max_deg": float(rot.max()),
            "gripper_max_abs_m": float(np.abs(actual[:, 6] - expected[:, 6]).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "settings.json").read_text())
    sys.path.insert(0, str(Path(cfg["pts_snapshot"]) / "src"))
    from pts.offline import CONDITIONS, RATES, delayed_replay
    fixed = np.load(cfg["fixed_inputs"], allow_pickle=False)
    saved = {c: np.load(run / "paired" / c / "outputs.npz", allow_pickle=False) for c in CONDITIONS}
    timing = {c: [json.loads(x) for x in (run / "paired" / c / "timings.jsonl").read_text().splitlines()] for c in CONDITIONS}
    summaries = {c: json.loads((run / "paired" / c / "summary.json").read_text()) for c in CONDITIONS}
    memory = {c: json.loads((run / c / "summary.json").read_text()) for c in CONDITIONS}
    for c, summary in memory.items():
        if summary["max_simultaneous_gpu_processes"] != 1 or len(summary["nvml_active_pids"]) != 1 or summary["memory_sampling_errors"]:
            raise AssertionError(f"GPU memory measurement is not isolated: {c}")
    baseline, one = saved["baseline_10step"], saved["pts_1x"]
    n, quality, checks, replay_summaries = len(cfg["frames"]), {}, {}, []
    replay_dir = run / "replay"
    replay_dir.mkdir(exist_ok=False)
    for c in CONDITIONS:
        data = saved[c]
        deterministic = all(np.array_equal(data["raw_actions"][j], data["raw_actions"][j % n])
                            for j in range(len(data["raw_actions"])))
        checks[c + "_repeat_exact"] = deterministic
        if c != "baseline_10step":
            checks[c + "_same_one_step_predictions"] = bool(np.array_equal(data["raw_actions"], one["raw_actions"]))
        per_obs, own_all, base_all, demo_all, actual_all = [], [], [], [], []
        for i, frame in enumerate(cfg["frames"]):
            count = int(data["lengths"][i])
            action, progress = data["executed_actions"][i, :count], data["progress"][i, :count]
            state = fixed["states"][i]
            own_ref = pose_at(state, data["raw_actions"][i], progress)
            base_ref = pose_at(state, baseline["raw_actions"][i], progress)
            demo_ref = pose_at(state, fixed["demonstration_actions"][i], progress)
            actual_all.append(action); own_all.append(own_ref); base_all.append(base_ref); demo_all.append(demo_ref)
            poses = np.vstack((state, action))
            translation = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) * 1000
            rotations = Rotation.from_rotvec(poses[:, 3:6])
            angle = (rotations[1:] * rotations[:-1].inv()).magnitude() * 1000
            own_error = errors(action, own_ref)
            checks[f"{c}_frame_{frame}_path_conservation"] = (
                own_error["position_max_mm"] < 1e-8 and own_error["rotation_max_deg"] < 1e-8
                and own_error["gripper_max_abs_m"] < 1e-10)
            per_obs.append({"frame": frame, "output_length": count, "progress": progress.tolist(),
                "rates": timing[c][i]["pts"]["rates"], "reasons": timing[c][i]["pts"]["reasons"],
                "vs_own_decoder_path": own_error, "vs_baseline_at_progress": errors(action, base_ref),
                "vs_demonstration_at_progress": errors(action, demo_ref),
                "max_translation_step_mm": float(translation.max()), "max_rotation_step_mrad": float(angle.max()),
                "steps_over_35mm": int(np.count_nonzero(translation > 35 + 1e-8)),
                "steps_over_60mrad": int(np.count_nonzero(angle > 60 + 1e-8))})
            d = max(1, math.ceil(summaries[c]["timing_ms"]["inference_ms"]["median"] * cfg["action_hz"] / 1000))
            for label, delays in (("measured_delay", (d,)), ("stall_stress", (d, d + 1, 40))):
                # Fixed H=50 predictions are reused as a scheduling fixture. Widely
                # separated dataset frames are NOT interpreted as a robot rollout.
                try:
                    replay = delayed_replay(data["raw_actions"][i], state, cfg, rate=RATES[c], delays=delays, ticks=150)
                    summary = {"condition": c, "frame": frame, "scenario": label, "status": "completed", **replay["summary"]}
                    checks[f"{c}_{frame}_{label}_continuation"] = replay["summary"]["max_entry_error"] < 1e-9
                    (replay_dir / f"{c}_frame{frame}_{label}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in replay["events"]))
                except ValueError as exc:
                    summary = {"condition": c, "frame": frame, "scenario": label,
                               "status": "blocked_by_diagnostic_bound", "reason": str(exc)}
                replay_summaries.append(summary)
        quality[c] = {"per_observation": per_obs,
                      "aggregate_vs_own_decoder_path": errors(np.vstack(actual_all), np.vstack(own_all)),
                      "aggregate_vs_baseline_at_progress": errors(np.vstack(actual_all), np.vstack(base_all)),
                      "aggregate_vs_demonstration_at_progress": errors(np.vstack(actual_all), np.vstack(demo_all))}
    decoder_change = errors(one["raw_actions"][:n].reshape(-1, 7), baseline["raw_actions"][:n].reshape(-1, 7))
    decoder_prefix_change = errors(one["raw_actions"][:n, :8].reshape(-1, 7), baseline["raw_actions"][:n, :8].reshape(-1, 7))
    old = np.load(Path(cfg["source_freespeed_run"]) / "paired/baseline/outputs.npz", allow_pickle=False)
    old_difference = errors(baseline["raw_actions"][:n].reshape(-1, 7), old["raw_actions"][:n].reshape(-1, 7))
    cross_process = {}
    for c in CONDITIONS:
        isolated = np.load(run / c / "outputs.npz", allow_pickle=False)
        cross_process[c] = errors(isolated["raw_actions"][:n].reshape(-1, 7), saved[c]["raw_actions"][:n].reshape(-1, 7))
    checkpoint_unchanged = json.loads((run / "checkpoint_before.json").read_text()) == json.loads((run / "checkpoint_after.json").read_text())
    checks["checkpoint_unchanged"] = checkpoint_unchanged
    checks["inputs_identical_to_freespeed"] = hashlib.sha256(Path(cfg["fixed_inputs"]).read_bytes()).hexdigest() == cfg["fixed_inputs_sha256"]
    for c in CONDITIONS:
        if c != "baseline_10step":
            checks[c + "_postprocessing_unchanged"] = summaries[c]["wrapper_equals_original_zero_noise_policy"]
    passed = all(checks.values())
    report = {"checks_passed": passed, "checks": checks, "quality": quality,
              "one_step_vs_10step_full_horizon": decoder_change, "one_step_vs_10step_first8": decoder_prefix_change,
              "baseline_vs_previous_freespeed_process": old_difference,
              "isolated_vs_paired_output_variation": cross_process, "paired_summaries": summaries,
              "isolated_memory_summaries": memory, "scheduler_replays": replay_summaries,
              "new_checkpoint_bytes": 0}
    write(run / "comparison.json", report)
    labels = {"baseline_10step": "Baseline 10-step", "pts_1x": "PTS 1-step 1x", "pts_1p5x": "PTS 1-step 1.5x", "pts_2x": "PTS 1-step 2x"}
    infer_ratio = summaries["baseline_10step"]["timing_ms"]["inference_ms"]["median"] / summaries["pts_1x"]["timing_ms"]["inference_ms"]["median"]
    completed = [r for r in replay_summaries if r["status"] == "completed"]
    lines = ["# Polish π0.5 / PTS 固定输入离线实验", "",
        f"完成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。实现/一致性检查：{'通过' if passed else '存在失败，见 comparison.json'}。", "",
        "本轮直接复用 FreeSpeed 的只读 checkpoint，新增 checkpoint 字节数为 **0**；未训练、导出或复制模型权重。", "",
        "## 对齐的实验条件", "",
        f"- 既有 checkpoint：`{cfg['checkpoint']}`。",
        f"- 输入和 OpenPI 源码来自 `{cfg['source_freespeed_run']}`；通过符号链接复用冻结观测、tokenizer 缓存和源码，PTS 自身保存源码快照及哈希。",
        f"- OpenPI `{cfg['source_version']['head']}` 加已保存工作区 diff；此版本不是已确认的原训练 commit。",
        f"- polish v4 训练集 episode 0，frame {cfg['frames']}，双相机 RGB 224×224，prompt `polish the curve`，15 Hz，batch 1。",
        "- 原有 OpenPI/forcevla 环境，H20 GPU 0。每组预热 15 次、测量 100 次；主要耗时/动作比较在同一个加载的模型中交错执行，独立进程另测各组显存。",
        "- 基线沿用 FreeSpeed 的固定高斯噪声和 10 步采样；PTS 使用完整 50×32 零 latent 和单步采样。这项解码改变是 PTS 方法的一部分，不能要求其预测与 10 步基线完全一致。",
        "- 模型预测 H=50、重规划阈值 s=8、倍率网格 0.25；1.5× 达到阈值时可能已消耗 9 步。全部几何误差按实际进度对齐。1.5× 为论文 π0.5 的参考 cap，2× 是与前次实验呼应的附加检查，未做任务开发集选参。",
        "- 固定输入前缀使用 d=2；夹爪窗口按 15 Hz 取 2–9 步，阈值 0.02，无指速反馈时保持完整窗口。",
        "- 35 mm / 60 mrad 是与 FreeSpeed 离线检查一致的诊断界限，**不是实测 Flexiv 执行器 box**；不能直接作为真机加速配置。", "",
        "## GPU 耗时与显存", "",
        "推理时间包含原输入处理、模型、同步返回 CPU 和输出后处理。PTS 后处理列包含构建路径以及生成达到重规划阈值的整段前缀；真机实现会把这些调度决策分散到控制步。此总时间不包含相机、网络、机器人或任务执行时间。", "",
        "| 条件 | 推理 median ms | PTS/前缀 median ms | 总 median ms | 总 p95 ms | 稳态驻留峰值 GiB | JAX 全程峰值 GiB | 输出个数 |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in CONDITIONS:
        t, m = summaries[c]["timing_ms"], memory[c]
        lines.append(f"| {c} | {t['inference_ms']['median']:.3f} | {t['postprocess_ms']['median']:.3f} | {t['total_ms']['median']:.3f} | {t['total_ms']['p95']:.3f} | {m['nvml_total_process_peak_bytes']['measure']/2**30:.3f} | {m['jax_memory_after_measurement']['peak_bytes_in_use']/2**30:.3f} | {summaries[c]['output_lengths']} |")
    lines += ["", f"同进程解码 median 比值（10 步 / 单步 1×）为 **{infer_ratio:.3f}×**，仅代表本组输入的推理加速。",
        "NVML 每 20 ms 采样，确认每个独立测量仅一个 GPU 进程；驻留分配与 JAX 活跃内存峰值口径不同。独立进程分配布局也会变化，微小显存差不能直接归因于倍率。", "",
        "## 解码变化与路径保持", "",
        f"相同输入在同一编译实例内重复推理、PTS 各倍率的原始预测一致性、原输出后处理一致性及路径保持检查：{'全部通过' if passed else '请检查失败项'}。",
        f"单步与 10 步预测的前 8 步平均位置差 **{decoder_prefix_change['position_mean_mm']:.3f} mm**、姿态差 **{decoder_prefix_change['rotation_mean_deg']:.3f}°**；完整 50 步平均位置差 {decoder_change['position_mean_mm']:.3f} mm。",
        "PTS 1× 保持其单步解码路径；与 10 步 baseline 的差异来自解码，不能误判为时间缩放误差。下表为按实际路径进度对齐的离线代理指标。", "",
        "| 条件 | 相对自身解码路径最大误差 mm | 相对10步基线平均差 mm | 相对示范平均差 mm | 最大单步平移 mm | 最大单步旋转 mrad |",
        "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        q = quality[c]
        lines.append(f"| {c} | {q['aggregate_vs_own_decoder_path']['position_max_mm']:.3g} | {q['aggregate_vs_baseline_at_progress']['position_mean_mm']:.3f} | {q['aggregate_vs_demonstration_at_progress']['position_mean_mm']:.3f} | {max(v['max_translation_step_mm'] for v in q['per_observation']):.3f} | {max(v['max_rotation_step_mrad'] for v in q['per_observation']):.3f} |")
    lines += ["", f"本轮 10 步基线相对前次 FreeSpeed 进程的最大位置差为 {old_difference['position_max_mm']:.6f} mm；跨进程变化另存 comparison.json，不把它归因于 PTS。", "",
        "## 异步接续检查", "",
        f"使用保存的真实模型 chunk 构造缓存调度回放，共 {len(replay_summaries)} 个场景，{len(completed)} 个完成。每场景 150 个控制步，固定延迟由实测推理解码 median 换算（不含整段前缀生成、传感器或网络）；附加 `(d, d+1, 40)` 步延迟用于模拟突发停顿。",
        f"完成场景中最大入口误差为 {max((r['max_entry_error'] for r in completed), default=0):.3g} 路径步；验证每次切换入口等于采集后累计执行倍率、最多一个推理请求、推理中继续执行，以及 chunk 耗尽时保持。",
        "缓存 chunk 不随模拟状态重算；五个相隔很远的示范帧也没有被拼成闭环轨迹。因此该部分只验证 R1 调度与退化处理，不是物理仿真、任务回放或真实速度测试。若完整路径触发诊断界限，会在 comparison.json 中记录阻止原因，不放宽限幅。", "",
        "## 结论范围与复跑", "",
        f"在相同的前 8 个目标位置上，10 步基线的平均示范位置误差为 {quality['baseline_10step']['aggregate_vs_demonstration_at_progress']['position_mean_mm']:.3f} mm，PTS 单步 1× 为 {quality['pts_1x']['aggregate_vs_demonstration_at_progress']['position_mean_mm']:.3f} mm。这直接反映本次解码改变的离线精度差异；R2 的路径保持并不会消除这项差异。",
        "当前只覆盖单条训练示范的五个观测。输出更短、推理更快、离线示范误差更小均不等同于打磨任务更快或质量更高；成功率、真实跟踪、接触力和表面质量仍需同 checkpoint 的真机对照。",
        "权重只读状态、文件数/大小/mtime/inode 在运行前后匹配；沿用原归档的逐文件 SHA-256 清单，本轮未重复读取全量 44.75 GB 做新哈希。结果目录没有模型参数副本。", "",
        "```bash", f"bash {run / 'reproduce.sh'} /mnt/work/shidong/results/pts/polish_repeat_01", "```", "",
        "输出目录必须不存在，GPU 0 须空闲；复跑需要保留 FreeSpeed 冻结实验目录及原 checkpoint。",
        "`paired/` 保存主要推理/动作样本及逐次计时；各条件顶层目录保存独立显存测量；`replay/` 保存接续事件；`comparison.json` 保存逐帧误差；`comparison.png/pdf` 保存图表；`checkpoint_before/after.json` 保存权重不变的检查。", ""]
    (run / "REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    write(run / "analysis_environment.json", {"python": sys.version, "executable": sys.executable,
          "numpy": np.__version__, "scipy_rotation": "SO3 geodesic", "matplotlib": matplotlib.__version__})
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.8, 1, 1))
    latency = fig.add_subplot(grid[0, :2])
    latency.boxplot([[r["total_ms"] for r in timing[c]] for c in CONDITIONS],
                    tick_labels=[labels[c] for c in CONDITIONS], showfliers=True)
    latency.set_ylabel("Synchronized inference + prefix processing (ms)")
    latency.set_title("100 measurements per condition; same loaded policy, rotated order")
    latency.grid(axis="y", alpha=.25)
    axm = fig.add_subplot(grid[0, 2])
    axm.bar(["10-step", "PTS 1x", "PTS 1.5x", "PTS 2x"], [memory[c]["nvml_total_process_peak_bytes"]["measure"]/2**30 for c in CONDITIONS])
    axm.set_ylabel("Process resident peak (GiB)")
    axm.set_title("Isolated-process NVML, 20 ms sampling")
    for i, frame in enumerate(cfg["frames"]):
        ax = fig.add_subplot(grid[1 + i // 3, i % 3], projection="3d")
        state = fixed["states"][i]
        end = math.ceil(max(saved[c]["progress"][i].max() for c in CONDITIONS))
        for raw, label, color, style in ((baseline["raw_actions"][i], "10-step path", "C0", "-"),
                                        (one["raw_actions"][i], "1-step path", "C1", "-"),
                                        (fixed["demonstration_actions"][i], "Demonstration", "gray", "--")):
            xyz = (np.vstack((state, raw[:end]))[:, :3] - state[:3]) * 1000
            ax.plot(*xyz.T, style, color=color, label=label, linewidth=1.5)
        for c, marker, color in (("pts_1p5x", "o", "C2"), ("pts_2x", "x", "C3")):
            count = saved[c]["lengths"][i]
            xyz = (saved[c]["executed_actions"][i, :count, :3] - state[:3]) * 1000
            ax.scatter(*xyz.T, marker=marker, color=color, s=22, label=labels[c])
        ax.set_title(f"Frame {frame}, progress 0..{end}")
        ax.set_xlabel("dx (mm)"); ax.set_ylabel("dy (mm)"); ax.set_zlabel("dz (mm)")
        if i == 0:
            ax.legend(fontsize=7)
    note = fig.add_subplot(grid[2, 2]); note.axis("off")
    note.text(0, 0.9, "One checkpoint, no weight copies\n\n"
              "10-step baseline: fixed Gaussian noise\nPTS: zero noise, one Euler step\n\n"
              "H=50, replan threshold s=8\n1.5x may consume 9 steps before replan\n"
              "Markers lie on the one-step path.\n\n"
              "Five fixed training observations only.\nNo robot success, force or surface-quality test.\n"
              "35 mm / 60 mrad are diagnostic bounds,\nnot a measured robot actuator box.", va="top", fontsize=11, linespacing=1.5)
    fig.suptitle("Polish v4 / checkpoint 49999: offline PTS decoder and execution checks")
    fig.savefig(run / "comparison.png", dpi=160); fig.savefig(run / "comparison.pdf"); plt.close(fig)
    write(run / "ARTIFACT_SHA256.json", {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(run.rglob("*")) if p.is_file() and not p.is_symlink() and p.name != "ARTIFACT_SHA256.json"})
    print(json.dumps({"checks_passed": passed, "report": str(run / "REPORT.md"), "inference_ratio": infer_ratio}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
