# 本地验证记录

日期：2026-11-16（UTC）。这是实现验证记录，不是论文性能复现报告。

**环境复用**：`/mnt/work/shidong/repos/openpi/.venv/bin/python`，Python 3.11.15；该 venv 使用 `/mnt/work/shidong/envs/forcevla` 的系统包。本次没有创建 conda 环境或安装/升级依赖。

| 组件 | 本次版本/来源 |
| --- | --- |
| JAX / jaxlib | 0.5.3 / 0.5.3 |
| PyTorch | 2.7.1 |
| NumPy / SciPy | 1.26.4 / 1.15.3 |
| 推理 GPU | NVIDIA H20，GPU 0；仅本次 smoke 使用 |
| OpenPI HEAD | `87707a1ee8947ec3ad3989450cd7bb62bc6392f7` |
| ForceBenchWeb HEAD | `3ac8742c59a85ccd4aec8d8c3c11a0c8c4486178` |

两个依赖仓库原本都存在本地修改，所以上述 commit 不是完整的环境快照。PTS 加载当前工作区版本，没有修改其源码。

验证内容：

- `bash scripts/python.sh -m pytest -q`：27 项通过，覆盖数值核心、配置契约、异步控制、异常停止、校准/指标及 OpenPI→ForceBench 本地 WebSocket 互通。
- `bash scripts/python.sh -m pts.run --mock --duration 3 --session mock_validation_01`：正常 finalize，生成两路 MP4、状态/控制 HDF5、视频时间戳和 PTS 日志。合成策略不涉及模型质量或物理跟踪。
- `pts.smoke_model`：加载本地 `pi05_full_peg_in_hole_sim_200_v1_15hz` 的 `50000` checkpoint，使用两张黑图、零 TCP 状态及 peg prompt。输出 `(50, 7)`，有限；重复推理逐元素相同；与同一原 OpenPI Policy 的零噪声调用逐元素相同。

另一次 5 秒完整试运行 `artifacts/mock_validation_final/` 完成 9 次 chunk 接续，使用 1.5×；保存 871 条状态、621 条控制/策略记录，两路 H.264 视频各 152 帧。其 `verification.json` 记录了实际检查结果。该历史试运行在时限停止时将内部取消记为 `CancelledError` 事件，资源清理成功；随后修正了事件标签，正常停止现记录为 `cancelled`，不再记为 `error`。

模型 smoke 原始结果在 [artifacts/model_smoke.json](artifacts/model_smoke.json)。单次首次调用包含编译约 11.64 秒，预热后单次调用约 81.34 毫秒。这不是多次延迟基准，也不包含网络、相机和机器人 SDK。该 checkpoint 使用仿真数据训练，不能代替真机 checkpoint 的验证。

首次模型检查发现默认 tokenizer 缓存为空、外部下载证书校验失败；已改为复用存在的 `/mnt/work/shidong/models/openpi` 缓存，随后检查通过。没有关闭 TLS 校验，也没有改动 OpenPI 下载器。

尚未验证：真实机器人连接、夹爪标定、接触任务、实际跟踪误差、真实成功率和吞吐量；当前计算节点没有 easyrobot、flexivrdk、pyrealsense2。真机端复用已有 `forcebench-runtime`，用匹配的真机 checkpoint 及本机校准参数完成这些验证。

## 2026-11-18：polish v4 checkpoint 离线实验

复用 FreeSpeed 已归档的 `polish_curve_v4_15hz/49999`、原 OpenPI/forcevla 环境、冻结的 OpenPI 源码和五个真实训练观测；未复制、转换或导出 checkpoint。权重共 46 个文件、44,745,781,897 字节，运行前后 size/mtime/inode/只读权限不变，新增权重字节为 0。

结果：[REPORT.md](/mnt/work/shidong/results/pts/20261118_polish_v4_49999/REPORT.md)，目录 `/mnt/work/shidong/results/pts/20261118_polish_v4_49999/`。

- 31 项自动化测试通过，包括新增的非整数重规划阈值、带延迟的接续和耗尽保持。
- 4 个条件每组预热 15 次、测量 100 次：10 步基线、PTS 单步 1×/1.5×/2×。主要 400 次测量在同一模型内交错执行；另有 400 次独立进程测量用于显存。
- 主要计时已将 NPZ 状态读取移到计时外；初次主对照仅保留在 `attempts/paired_with_npz_lookup/`，不用于最终结论。显存测量与主对照使用的脚本版本由 `code_provenance.json` 标记；模型及 PTS 算法相同。
- 最终推理 median：10 步 82.591 ms，PTS 单步约 60.46 ms；计入整段前缀生成后，PTS 1×/1.5×/2× 分别为 64.407/69.882/70.746 ms。稳态进程显存均约 8.4 GiB。
- 重复推理精确一致，PTS 各倍率原始单步预测精确一致；路径保持数值误差小于 `1e-8 mm`；40 个缓存 chunk 接续场景各执行 150 步，全部通过，入口进度误差为 0。
- 单步与 10 步前 8 个目标平均位置差 2.785 mm；同一进度的示范位置误差从 1.381 mm 变为 2.611 mm。因此本轮验证了实现与推理加速，并未证明打磨精度改善。

PNG/PDF 图表已渲染并检查。原始动作、逐次计时、显存采样、接续事件、源码/输入哈希、权重不变记录及复跑脚本均保存。缓存调度回放不包含物理动力学或状态相关的新观测；真实接触力、表面质量、任务成功率和任务级加速仍未测量。


## 2026-11-20：flip / polish 统一真机入口

本次只执行 CPU/模拟验证，没有打开真实机器人/相机，没有加载 checkpoint 参数到 GPU，也未进行硬件标定。

- 两个任务的真实 OpenPI 训练配置、TCP 输入/输出变换和 checkpoint 自带 norm_stats 经 `pts.server --inspect` 检查通过；只在进程内补充元数据，共享训练源码与 checkpoint 未改动。
- FreeSpeed / flip：2 秒模拟超时试次，结果 timeout / false，349 条状态、200 条控制/策略记录，global/wrist 视频均正常生成。
- PTS / polish：2 秒模拟超时试次，结果 timeout / false，347 条状态、254 条控制/策略记录，双相机视频正常生成。
- FreeSpeed / polish：通过真实 CLI 发成功停止信号，2.568 秒后停止，452 条状态、255 条控制/策略记录；双相机各 78 帧采集记录。
- PTS / flip：通过真实 CLI 发成功停止信号，2.996 秒后停止，521 条状态、381 条控制/策略记录；global/wrist 各 87 帧采集记录。首次使用硬件同步容差的 mock 因初始化抖动被拒绝；随后只调整默认模拟模式容差，硬件限制未放宽。
- 上述 HDF5 的 complete 标志、时间戳顺序、位姿/力/原始预测字段及视频 manifest 已核验。成功信号仅为模拟接口测试，不代表实际任务成功。
- 成功试次经统一 exporter 导出 CSV，再经 metrics 汇总；按 task/mode/method 分组。超时、待评分、重复试次、旧控制 token、清理失败和取消在自动化测试中覆盖。
- 自动化测试：PTS 43 项通过。

## 2026-11-20：ForceBenchWeb 上游更新兼容

ForceBenchWeb 更新至 upstream `a8fe9b2` 后，polish 默认 run 配置改用
`polish_curve_v4_pi05.yaml`，任务名校验同步为 `polish_curve_v4`。
旧工作区中的图像裁剪和姿态分支扩展已与上游合并保留。

- PTS 自动化测试：43 项通过。
- polish 模拟超时试次：timeout / false；351 条独立状态、253 条控制/策略记录，双相机各 62 帧。
- flip 模拟显式成功停止：2.771 秒；485 条状态、352 条控制/策略记录，global/wrist 各 79 帧。
- HDF5 完成标记、时间戳、视频文件及评分导出/汇总通过检查。
- 本次没有连接机器人或加载真实模型；模拟成功信号只用于验证接口。
