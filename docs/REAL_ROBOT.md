# Flip / polish 真机试次

本说明对应 FreeSpeed / PTS 的 `trial-v1` 协议。二者使用同一任务计时、结果字段和 ForceBench 记录接口。真实设备操作仅在显式 `--mode real`（FreeSpeed）或 `--hardware`（PTS）时发生。

## 模型和工作区

控制机需要匹配的 ForceBench 源码和原机器人环境。

当前入口适配 ForceBenchWeb `a8fe9b2`（2026-11-11 的 upstream main）。polish 使用
`configs/runs/polish_curve_v4_pi05.yaml`，任务名为 `polish_curve_v4`；旧版
`polish_curve_pi05.yaml` 已被上游替换。更新控制端时同时更新 ForceBenchWeb。

默认工作区布局：

```text
workspace/
  ForceBenchWeb/
  FreeSpeed/
  PTS/
  openpi/
```

ForceBench 不在同级时分别设置 `FREESPEED_FORCEBENCH_ROOT` / `PTS_FORCEBENCH_ROOT`；这些变量同时影响 Python 导入和任务配置查找。GPU 端需要含下列训练配置的 OpenPI 版本，不能用同名但不同输入/动作定义的配置替代。

| 任务 | OpenPI 配置 | checkpoint 自带统计目录 |
| --- | --- | --- |
| flip | `pi05_flexiv_1arm_2cam_flip_fp2_v3_selected_fixed_15hz` | `assets/magikonki/flip_fp2_v3_selected_fixed_15hz/` |
| polish | `pi05_flexiv_1arm_2cam_polish_curve_v4_15hz` | `assets/magikonki/polish_curve_v4_15hz/` |

两者均为原始 π0.5 单任务 checkpoint `49999`，不是 RLT/FP2 权重。接口为双相机 RGB + 7D xyz/轴角/夹爪状态，模型输出经原逆变换恢复为 50×7 绝对 TCP/夹爪目标，训练动作时基 15 Hz。

服务入口核对训练数据类、repo_id、原始 Delta/Absolute TCP 变换、50×32 模型维度和 checkpoint 内的任务统计。仅对这两个已知配置在内存中补充 task、action、control_frequency_hz；未知配置不会被自动假设成 TCP。不会改写训练配置、checkpoint 或共享 OpenPI 源码。

## GPU 端

以下环境变量换成推理机实际路径；当前服务器对应源码为 `/mnt/work/shirun/repos/openpi`，已有解释器为 `/home/shidong/code/openpi/.venv/bin/python`。完整模型目录必须包含 `params/` 和 `assets/`。

```bash
cd /path/to/workspace/PTS
export PTS_OPENPI_ROOT=/path/to/matching/openpi
export PTS_PYTHON=/path/to/inference/environment/bin/python
export OPENPI_DATA_HOME=/path/to/openpi/cache
export TASK_CKPT=/path/to/flip_checkpoint/49999

# 只核验配置、变换和统计，不加载模型权重，不连接机器人。
bash scripts/python.sh -m pts.server --task flip --checkpoint "$TASK_CKPT" --inspect

# FreeSpeed / Sync-1x 使用原 10 步解码。
bash scripts/python.sh -m pts.server --task flip --checkpoint "$TASK_CKPT" \
  --decoder stochastic_10_step --port 8025
```

运行 PTS 主方法时，将服务端解码改为 `--decoder zero_noise_one_step`。运行 polish 时，同时将 `--task` 和 `TASK_CKPT` 改成 polish 对应项。客户端会在 ready pose 之前检查服务端 task、频率和解码器，任务混用会报错。FreeSpeed 可直接连接这个 10 步服务；控制机不需要导入 PTS 包。

## 控制端

先激活已部署机器人 SDK 的原控制环境。FreeSpeed 新入口与 PTS 一样，通过 ForceBench 执行配置中的 ready pose、夹爪准备和力传感器清零；这与旧 FreeSpeed 从当前位置直接开始的行为不同。

```bash
cd /path/to/workspace/FreeSpeed
python scripts/run_forcebench.py --task flip --mode inspect

# 同步原前缀基线；模型服务为 stochastic_10_step。
python scripts/run_forcebench.py --task flip --mode real \
  --method reference --rate 1 --server-url ws://GPU_HOST:8025 \
  --duration 60 --session flip_sync_s00_r00 --setup-id 0 --repeat-id 0

# FreeSpeed 的 1.5x / 2x：替换 method、rate 和唯一的 session。
python scripts/run_forcebench.py --task flip --mode real \
  --method freespeed --rate 1.5 --server-url ws://GPU_HOST:8025 \
  --duration 60 --session flip_fs15_s00_r00 --setup-id 0 --repeat-id 0
```

```bash
cd /path/to/workspace/PTS
export PTS_PYTHON="$(command -v python)"

# 服务端为 zero_noise_one_step。默认 rate_cap=1。
bash scripts/python.sh -m pts.run --task flip --hardware \
  --endpoint ws://GPU_HOST:8025 --duration 60 \
  --session flip_pts1_s00_r00 --setup-id 0 --repeat-id 0

# 用已采集的脉冲 CSV 建立该任务工具/负载下的执行器边界。
bash scripts/python.sh -m pts.calibrate measured_flip_pulses.csv \
  --base-config configs/flip_15hz.yaml --output configs/flip_calibrated.yaml --rate-cap 1.5

bash scripts/python.sh -m pts.run --task flip --hardware \
  --pts-config configs/flip_calibrated.yaml --endpoint ws://GPU_HOST:8025 \
  --duration 60 --session flip_pts15_s00_r00 --setup-id 0 --repeat-id 0
```

polish 使用 `--task polish`；PTS 标定基准文件换成 `configs/polish_15hz.yaml`。边界和夹爪参数必须适用于该工具/负载。`--rate-cap` 可覆盖倍率，但上限大于 1 时仍必须存在实测 actuator；`mock_15hz.yaml` 的合成边界不能当标定数据。PTS-K 需要服务端和控制端都指定 `--decoder stochastic_10_step`。

两个入口保留 15 Hz 动作时基、100 Hz 目标下发和 `trajectory_time_scale=1`。状态/力采样目标来自 ForceBench 的 `state_sample_frequency_hz`，现有任务配置为 200 Hz。实际频率受 SDK、线程调度等影响，以时间戳统计。FreeSpeed 的 `--cycles` 只是调试上限，正式比较使用共同的成功/失败判定和任务超时。

`--output` 在两个入口中均表示试次父目录；默认在各自 `artifacts/`，`--session` 为唯一子目录名。已有结果不会覆盖。默认任务为 flip；默认模式分别为 inspect 和 mock。

## 开始、停止和评分

1. 准备唯一结果目录，记录配置。
2. 用黑图和零状态预热模型，输出丢弃。
3. ForceBench 完成相机、机器人 ready pose 和控制器准备。
4. 执行器取得初始状态后，在首个任务观测前启动计时；首次真实模型推理计入耗时。
5. 收到显式成功/失败/待评分信号，或达到墙钟超时后取消执行、停止设备，完成视频/HDF5 写入。

任务时间终点为停止判定时刻；设备清理/视频编码另记为 `finalization_s`。超时检查以 10 ms 轮询执行，不保证硬实时。视频/状态覆盖范围可能略宽于任务计时，分析时按 `started_monotonic_s` 和 `stopped_monotonic_s` 裁剪。Ctrl+C/SIGTERM 记为 aborted；未开始的准备失败没有任务耗时，不属于已执行试次。

成功不能由机器人走完若干步推断。默认提供人工/外部评估接口，没有虚构物体视觉判定器。测试前用 `--criteria` 记录冻结的评分规范版本和数值阈值：

- `manual_flip_v1`：指定目标面和容许姿态/位置范围、稳定时长、掉落/滑出定义；由操作者或外部判定器确认。
- `manual_polish_v1`：指定路径覆盖、接触连续性、力范围及表面质量验收条件。若需要事后表面测量，完成运动时先停止为 unscored，再补评分。

在第二个终端、相应项目目录执行（外部判定器也可调用同一函数/CLI）：

```bash
# FreeSpeed 已达到任务成功条件。
python -m freespeed.trial signal artifacts/flip_fs15_s00_r00 \
  --outcome success --reason '目标面稳定达到已登记标准' --evaluator operator-A

# PTS 执行失败。
bash scripts/python.sh -m pts.trial signal artifacts/flip_pts15_s00_r00 \
  --outcome failure --reason '物体滑出目标区' --evaluator operator-A

# polish 完成运动，等待表面质量验收。
bash scripts/python.sh -m pts.trial signal artifacts/polish_pts15_s00_r00 \
  --outcome unscored --reason '轨迹完成，待表面检测' --evaluator operator-A

# 完成停止和文件写入后，给出质量判定；0 表示失败。
bash scripts/python.sh -m pts.trial score artifacts/polish_pts15_s00_r00 \
  --success 1 --reason '通过预登记的覆盖、接触力及表面质量标准' --evaluator reviewer-A
```

`success` 是判定器显式报告的成功；超时、报错和中断不会自动算成功。待评分结果不可直接汇总。评分保存独立的 `trial_score.json`，不改写运行事实；已有评分不可覆盖，超时/失败/报错不可改判成功。可将初步成功在事后质量检查中降为失败。

手写控制文件时必须使用该次 `trial_result.json` 中的 `control_token`，以原子重命名发布 `trial_control.json`，字段为 `control_token/outcome/reason/evaluator`。推荐使用 CLI，它会拒绝尚未开始或已经结束的试次。

## 数据与汇总

| 文件 | 内容 |
| --- | --- |
| `trial_result.json` | 任务、方法、模式、布置/重复号、评分规范、开始/停止时间、结果、原因、判定者、耗时 |
| `trial_score.json` | 可选的事后质量验收结论，不覆盖运行记录 |
| `lowdim.h5` | 独立采样的实际 TCP、关节、夹爪、六维力及时间戳 |
| `control.h5` | 下发目标、完整策略预测、推理延迟及时间戳 |
| `global.mp4` / `wrist.mp4` | 双相机录像；平台定义额外相机时一并记录 |
| `video_timestamps.jsonl` / `video_manifest.json` | 相机时间戳、视频文件和帧数 |
| `run_manifest.json` / `runtime_metadata.json` | ForceBench 任务与设备配置 |
| `freespeed_config.json` / `events.jsonl` / `chunk_*.npz` | FreeSpeed 参数、CI/f/M、采样索引、原始/处理后动作 |
| `pts_config.json` / `pts_events.jsonl` | PTS 配置、请求/接管进度、倍率、限速原因、等待 |

正式评分以 `trial_result.json` 加可选 `trial_score.json` 为准。FreeSpeed `summary.json` 仍是算法循环诊断，不作为成功判定。若视频/HDF5 最终检查失败，统一试次结果降为 error，保留已发生的运行耗时。

```bash
# 在 PTS 目录，可混合导出两个项目的结果。
bash scripts/python.sh -m pts.trial export \
  /path/to/FreeSpeed/artifacts/flip_fs15_s00_r00 \
  /path/to/PTS/artifacts/flip_pts15_s00_r00 --output scored_trials.csv
bash scripts/python.sh -m pts.metrics scored_trials.csv
```

FreeSpeed 提供等价的 `python -m freespeed.trial export` 和 `python -m freespeed.metrics`。导出会拒绝重复、未结束或待评分试次，不会静默丢弃失败。汇总按 task/mode/method 分组，避免把 mock、硬件或两个任务混在一起。`control_steps=duration_s*15` 表示含等待的名义时间步，不是 100 Hz 下发次数。所有已执行尝试的时间均进入吞吐量分母。复位耗时与表面测量耗时需额外记录，当前任务吞吐量不包含它们。

## 无硬件验证

```bash
# FreeSpeed 目录
python scripts/run_forcebench.py --task flip --mode mock --duration 3 --session check_flip
python scripts/run_forcebench.py --task polish --mode mock --duration 3 --session check_polish
python -m pytest

# PTS 目录
bash scripts/python.sh -m pts.run --task flip --mock --duration 3 --session check_flip
bash scripts/python.sh -m pts.run --task polish --mock --duration 3 --session check_polish
bash scripts/python.sh -m pytest -q
```

PTS 默认 mock 的输入年龄/响应年龄/状态相机偏差容差为 1.0/2.0/0.2 秒，以覆盖合成相机和视频初始化抖动；有效设置写入 pts_config.json。硬件配置保持原严格检查。

没有显式成功信号时，上述 mock 到时正常清理、结果为 timeout/false，这是预期行为。mock 验证记录和调度接口，不证明真实任务成功、标定或硬件时序已通过。
