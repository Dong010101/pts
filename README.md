# PTS：基于现有 OpenPI / ForceBench 的独立复现

对应论文 [Same Path, Less Time: Training-Free Fast Execution of Vision-Language-Action Models under Inference Latency](https://arxiv.org/abs/2611.01390v1)。本目录按正文 R1–R4 和附录 B 实现，不是作者发布的官方代码。目录与 `LaWAM`、`openpi`、`ForceBenchWeb` 并列。

**当前交付：可运行的 PTS 核心、π0.5 服务端、ForceBench 真机适配和离线验证。未执行真机实验，不能据此声称复现了论文成功率或加速比。** 论文评测使用 LIBERO 相对增量控制；当前真机模型输出绝对 TCP 位姿，本实现明确保留了这项适配差异。

## 环境与入口

不新建 conda 环境，也不升级现有依赖。`scripts/python.sh` 通过 `PYTHONPATH` 加载本目录及相邻项目：

| 进程 | 复用环境 | 依赖/职责 |
| --- | --- | --- |
| 推理服务 | `../openpi/.venv/bin/python` | 已有 OpenPI、JAX/PyTorch、CUDA、模型权重和归一化统计 |
| 真机控制 | 机器人机器上的原 ForceBench 环境 | NumPy、SciPy、PyYAML、websockets、HDF5、视频编码，以及已部署的 easyrobot、flexivrdk、pyrealsense2 |
| 本机 mock / 单元测试 | OpenPI 环境 | 复用本机可用的 ForceBench、NumPy/SciPy、pytest 等 |

控制端只导入 PTS 数值核心和 ForceBench，不需要安装 OpenPI 模型依赖。两个进程可以在不同机器，使用 OpenPI 原有 WebSocket 协议通信。当前计算节点未安装三项真机 SDK；在已有机器人控制环境中执行硬件命令。

可覆盖路径：`PTS_PYTHON` 指定已有 Python，`PTS_FORCEBENCH_ROOT` 指定 ForceBench 源码根目录，`PTS_OPENPI_ROOT` 指定 OpenPI 根目录。脚本在发现本机 `../models/openpi/big_vision/paligemma_tokenizer.model` 缓存时设置 `OPENPI_DATA_HOME`；用户显式设置的值优先。

```bash
cd /mnt/work/shidong/repos/PTS
bash scripts/python.sh -m pytest -q
bash scripts/python.sh -m pts.run --mock --duration 10
```

`--mock` 使用合成策略、机器人、相机；`--dry-run` 使用真实推理服务与模拟硬件；只有显式 `--hardware` 才打开真机设备。每次运行创建独立的 `artifacts/pts_时间戳/`，不会覆盖已有 trial。

## 真机流程

1. 在推理机启动 PTS 服务，传入与真机数据、动作频率匹配的训练配置及 checkpoint。
2. 在控制机指定原 ForceBench 配置和环境。程序先验证服务元数据，用黑图和零状态预热模型并丢弃输出；随后进入原有 ready pose、相机和控制器初始化流程。
3. 先使用 `robot_15hz.yaml` 的异步单步 1× 配置检查任务行为。测定执行器线性范围、夹爪时长和传感器时间对齐后，生成校准配置，再比较 1× 与加速版本。
4. 在开发场景选择 rate cap，冻结配置后，在相同测试场景评估成功率和所有尝试的总耗时。

推理机示例，`PTS_CKPT` 必须换成实际的真机 checkpoint：

```bash
export PTS_CKPT=/absolute/path/to/real_robot_checkpoint
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bash scripts/python.sh -m pts.server \
  --config pi05_full_peg_in_hole_z_15hz \
  --checkpoint "$PTS_CKPT" --host 0.0.0.0 --port 8025
```

若使用 LoRA 或其他任务，改成对应的训练 config；不要只改 checkpoint 而保留不匹配的 config。checkpoint 中需要包含模型参数和训练时的归一化 assets。服务元数据包含 decoder、积分步数、动作维度、horizon 和 checkpoint。

控制机示例，先在本机挂载或复制 PTS 目录，再使用已能运行真机的 Python：

```bash
conda activate forcebench-runtime
export PTS_PYTHON="$(command -v python)"
export PTS_FORCEBENCH_ROOT=/path/to/existing/ForceBench
bash scripts/python.sh -m pts.run --hardware \
  --run-config "$PTS_FORCEBENCH_ROOT/configs/runs/peg_in_hole_v2_200sets_15hz_pi05.yaml" \
  --pts-config configs/robot_15hz.yaml \
  --endpoint ws://INFERENCE_HOST:8025 --duration 30
```

`--hardware` 会执行现有配置中的 ready pose，随后运动；`Ctrl+C` / SIGTERM 通过原驱动执行停止和资源清理。`--duration` 是试验运行时间上限，不含预热及 ready 阶段。任务是否成功仍由实验评分决定，程序不会把正常退出记成任务成功。`--visualize` 可开启原 ForceBench 预测可视化。

当前接入点是独立 trial CLI，复用 ForceBench 的设备、预处理、动作解码、准备/停止、HDF5 和视频记录；尚未向原 Web UI 添加 PTS 方法选项。

## 实现与论文的对应关系

| 规则 | 文件 | 行为 |
| --- | --- | --- |
| R1 解码 | `src/pts/openpi_policy.py` | 原 OpenPI `create_trained_policy`，`num_steps=1`，完整 padded latent 置零，保留原归一化逆变换与后处理 |
| R1 接续 | `core.py`、`clock.py` | 一次最多一个请求，按已消耗路径步数触发；新 chunk 入口为从图像采集时刻到切换时刻累计的已发送进度 |
| R2 | `paths.py` | 对分段线性累计路径做差，支持分数步，不用单个 waypoint 乘倍率代替积分 |
| R3 | `core.py` | 夹爪指令变化后的窗口内使用 1×；指令取所消耗区间的最后动作 |
| R4 | `core.py` | 取 0.25 网格上的最大可行倍率，结合带符号平移 box、`(H-s)/(2d)` 与 cap；检查中间路径拐点，排除不连续的可行区间 |
| 真机执行 | `forcebench.py` | 后台推理，复用原驱动；按原路径各段生成高频目标，切换不使用 DTW |

论文参考参数在 `configs/paper_pi05.json`，使用相对增量接口及 LIBERO 单位。`IncrementPath` 接收已经过策略后处理的 `[H,7]` 动作；若接入 LIBERO，应先按论文将各原始动作裁剪至 `[-1,1]`，并在控制器层允许叠加后的动作通过。这里没有安装或接入 LIBERO，也没有把其 box 值写进真机配置。

真机适配的具体差异：

- **动作**：观测时 TCP 位姿作为路径起点，将模型给出的绝对目标转成逐段运动；平移用米，姿态以 SO(3) 的 SLERP 和相对旋转组合。新 chunk 的剩余运动接在最后发送的位姿上，因此与原模型逐个发送绝对目标的方式存在差异，需要实际任务验证。
- **时间**：保留已有训练动作频率 15 Hz，原控制配置约 100 Hz 下发中间位姿；在动作边界额外唤醒，避免每步取整导致系统性降至 14.3 Hz。Python/SDK 调度为尽力执行，超时会记录。`d` 使用图像采集到可切换时刻的实测年龄，取近期窗口最大值并设初始下界，而非固定注入延迟。
- **夹爪**：按秒指定窗口，并换算成动作步数；以夹爪开口宽度的差分估算单指速度，默认乘 0.5。缺失反馈时保留完整窗口。此估算与仿真直接读取两指速度不同；窗口及阈值需要本机测量。
- **限幅**：真机加速要求提供以 `metres_per_action_step` 为单位的 box；另受原驱动速度对应的每步平移范数/角路程约束。控制器原速度、加速度、接触力和已配置的工作空间限制继续生效。这些不能代替负载/接触状态下的跟踪验证，也不保证物理轨迹严格保形。
- **边界情况**：剩余不足一步时走完最后分数步；无可用 chunk 时保持。若延迟可行上界小于 1，则保持并请求新 chunk；若基础 1× 指令已超出 box，报错停止，避免用裁剪破坏累计路径。

`robot_15hz.yaml` 默认 `rate_cap: 1.0`，用于单步异步基线；完整加速逻辑已实现，但未经真机标定的 `mock_15hz.yaml` 不能用于硬件。

## 校准、对照和记录

`pts.calibrate` 只处理已采集 CSV，不发运动命令。使用同一负载、控制频率和控制器设置，从静止测量六个平移方向的多个指令水平，记录 nominal/actual 位移；输入格式：

```text
axis,command_m_per_step,nominal_m,measured_m
x+,0.001,0.010,0.0098
```

实际文件需包含 `x+ x- y+ y- z+ z-`，每个水平默认至少三次；负方向的 nominal 和 measured 使用负值。工具按水平从小到大保留连续通过区间，每次位移误差不得超过 5%，遇到失败或重复数不足即截断。它不会推断未测区间的动态跟踪能力，也不会自动选择任务倍率。

```bash
bash scripts/python.sh -m pts.calibrate measured_pulses.csv \
  --base-config configs/robot_15hz.yaml \
  --output configs/robot_calibrated.yaml --rate-cap 1.5
```

建议主对照为相同 checkpoint、单步 decoder、相同初始场景下的 `rate_cap=1` 和校准后的 PTS。这样可单独检查时间缩放的效果。附加对照可用 `stochastic_10_step`，须同时修改服务端 `--decoder` 和控制端 YAML 中的 `decoder`；保留 1× 或加速 cap 分别得到异步多步对照或 PTS-K。原 ForceBench 固定频率/DTW 执行器可以另做系统对照，但不是论文的同条件异步 1×。

每个 trial 保存：

- `pts_events.jsonl`：配置/服务元数据、请求、观测时进度、chunk 入口、倍率、限速原因、完成进度、延迟、退出及异常。
- `pts_config.json`、`pts_warmup.json`：本次 PTS 设置及预热信息；PTS 实际调度以这些文件和 events 为准，原 ForceBench manifest 保留其基础配置身份。
- `lowdim.h5`、`control.h5`、双相机 MP4/时间戳及原 ForceBench 元数据。

评分表格式为 `method,trial_id,success,duration_s,control_steps`，成功填 0/1，保留失败与超时试次：

```bash
bash scripts/python.sh -m pts.metrics scored_trials.csv
```

输出 pooled 成功率、每小时成功数、每 1000 控制步成功数和尝试耗时。`control_steps` 应统一使用名义动作时间步（本机 15 Hz），**不要用约 100 Hz 的驱动循环次数**；跨频率比较优先用真实秒数。论文排除每集第一次推理；本 CLI 则记录任务开始后的首次真实观测等待，黑图预热在任务计时之前。做论文口径比较时需对所有方法统一计时范围；真机 reset 成本也应单独记录并一致处理。

## 本地验证

测试覆盖累计路径守恒、非整数入口、变速接续、夹爪窗口、连续可行限速、SO(3) 组合、异步执行、异常/取消清理及原 OpenPI WebSocket 与 ForceBench 客户端互通。完整 mock trial 已验证 HDF5 与双相机视频生成。详见 [VALIDATION.md](VALIDATION.md)。

真实模型的无硬件检查：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bash scripts/python.sh -m pts.smoke_model \
  --config YOUR_MATCHING_OPENPI_CONFIG --checkpoint /absolute/path/to/checkpoint
```

该工具仅检查相同合成输入的重复输出、形状、有限值及原后处理一致性；不评估闭环任务质量。

## Polish v4：复用 FreeSpeed 的冻结实验

polish 实验使用既有的只读 `polish_curve_v4_15hz/49999` checkpoint，路径为：

```text
/mnt/work/shidong/models/openpi/checkpoints/pi05/pi05_flexiv_1arm_2cam_polish_curve_v4_15hz/polish_curve_v4_15hz/49999
```

不复制、转换或导出权重。真实观测、tokenizer 缓存和对应的 OpenPI 源码也直接引用 FreeSpeed 已冻结的实验。PTS 核心及实验脚本单独保存快照，模型参数引用原归档，没有在 PTS 下保存额外副本。

本轮结果目录：`/mnt/work/shidong/results/pts/20261118_polish_v4_49999/`，结果入口为 [REPORT.md](/mnt/work/shidong/results/pts/20261118_polish_v4_49999/REPORT.md)。该离线入口使用原 polish 模型的数据变换，不依赖当前插孔服务端的 config 名称或元数据约定。

实验沿用 episode 0 的 frame 50/200/350/500/650、双相机 224×224、seed 42、每条件预热 15 次和测量 100 次；同一模型内交错比较 10 步固定噪声基线与 PTS 零噪声单步 1×/1.5×/2×，另在独立进程测显存。原始预测、进度、动作、逐次耗时及显存样本均保存。PTS 的 1× 恒等检查针对它自己的单步预测；单步与 10 步的差异单独统计。

此处保持完整预测 H=50、重规划阈值 s=8，因此 1.5× 的首段可以消耗 9 个路径步。几何指标按实际进度对齐，而不把不同长度数组按序号直接相减。额外的缓存 chunk 延迟回放检查接续和耗尽保持；它不使用机器人动力学或状态相关的新观测，不构成闭环任务仿真。

重跑到新目录：

```bash
bash /mnt/work/shidong/results/pts/20261118_polish_v4_49999/reproduce.sh \
  /mnt/work/shidong/results/pts/polish_repeat_01
```

或者从当前 PTS 源码新建实验：

```bash
../openpi/.venv/bin/python scripts/run_polish_benchmark.py \
  --source-run /mnt/work/shidong/results/freespeed/20261118_202654_polish_v4_49999 \
  --output /mnt/work/shidong/results/pts/polish_new_experiment
```

结果目录必须不存在，GPU 0 必须空闲，需保留原 FreeSpeed 冻结实验目录。分析图表复用已有 `/usr/bin/python3` 的 matplotlib，可通过 `--analysis-python` 指定其他已有解释器。每次重跑仍引用原 checkpoint，不产生权重副本。

离线 35 mm / 60 mrad 界限只用于诊断，并非实测执行器标定；这五个输入来自训练示范。是否适合真机打磨仍需验证跟踪误差、接触力、成功率及表面质量。
