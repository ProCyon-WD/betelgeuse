# 训练耗时、内存与收敛曲线

统一入口现在为每次独立训练保存资源统计和逐轮实际训练损失。
覆盖 FedCyon、FixMatch、FedConsist、CBAFed、RSCFed、IsoFed，以及 ImageNet-100
的 FedPDM / SAGE-Pure，以及后续接入的 ImageNet-100 FixMatch/FedConsist/RSCFed、CIFAR-10/SVHN FedPDM/SAGE-Pure。观测功能本身不改变损失、采样、优化器或聚合公式。

## 输出

每个 run 额外生成三份文件：

```text
<metrics_dir>/<dataset>/<save_name>/run_<run>_training.json
<metrics_dir>/<dataset>/<save_name>/run_<run>_training.csv
<metrics_dir>/<dataset>/<save_name>/run_<run>_training.npz
```

例如 FedPDM 默认配置第一遍训练：

```text
outputs/metrics/imagenet100/fedpdm_imagenet100_cmc_label01/run_1_training.json
outputs/metrics/imagenet100/fedpdm_imagenet100_cmc_label01/run_1_training.csv
outputs/metrics/imagenet100/fedpdm_imagenet100_cmc_label01/run_1_training.npz
```

JSON 包含 run 的最终汇总和逐轮记录；CSV 和 NPZ 只保存逐轮数据，用于绘图。
每完成一轮就原子替换文件，包含该 run 到目前为止所有已完成轮次，并不是只保留最后一轮。
正常结束后 JSON 状态变为 `completed`；可捕获异常/键盘中断保存 `failed` / `interrupted`。
未完成的轮次单独记入 `interrupted_round`，不伪装成已完成的曲线点。
若流程正常返回但没有完成配置要求的轮次，状态是 `incomplete`。
进程被 SIGKILL、机器断电等情况不能执行收尾，保留此前已落盘结果，状态可能仍是 `running`。

各 run 单独命名，记录实际训练 seed；划分 seed 仍由原配置与数据摘要保存。
同名实验使用 `--overwrite` 重跑仍会覆盖同 run 的文件；需要保留实验时请更换 `save_name`。
这不是 checkpoint 功能，不能据此恢复模型或优化器。

## Training time

时间单位为秒，使用单调高精度时钟。CUDA 在训练阶段起止同步选定设备，避免只量到
异步 kernel 提交时间。

| 字段 | 定义 |
| --- | --- |
| JSON 顶层 `training_time_seconds` | 所有训练阶段耗时之和，包含 CBAFed 的初始有标签预训练；如有中断轮，也包含已用的部分训练时间 |
| JSON 顶层 `total_time_seconds` | 从创建服务器对象前到训练流程结束的总墙钟时间，含数据/客户端初始化、评估和周期性结果写盘，不含最后一次统计文件写盘与对象清理 |
| 逐轮 `training_time_seconds` | 该轮客户端选择、本地训练、模型聚合及原有轮内诊断耗时；不含初始化、全局评估和统计文件写盘 |
| 逐轮 `cumulative_training_time_seconds` | 截至该轮的累计训练阶段耗时，可作时间轴 |
| 逐轮 `elapsed_time_seconds` | 从本 run 开始到该轮训练完成的墙钟时间，包含此前的初始化、评估等 |

旧算法本来就计算的梯度方差等轮内诊断仍包含在训练阶段，未静默关闭。
因此比较纯算法计算效率时，应统一这些诊断设置。采样线程与记录操作也有少量观测开销。

CBAFed 正式通信轮次前的有标签预训练单独保存在 JSON 的 `warmup` 列表中，
记录其 loss、耗时和内存，并计入总训练时间及累计训练时间，但不占用通信轮次、
不混入 CSV/NPZ 的逐轮收敛曲线。其他方法的 `warmup` 列表为空。

## Memory footprint

全部内存字段使用 **MiB = 1024² bytes**，不是十进制 MB。
JSON 顶层 `memory_footprint` 是整个 run 的峰值，包含初始化、评估和训练；
CSV/NPZ 各行是该轮训练阶段的峰值。

| 字段 | 定义 |
| --- | --- |
| `cpu_peak_rss_mib` | 主 Python 进程的 RSS 峰值 |
| `cpu_tree_peak_rss_mib` | 主进程及其递归子进程 RSS 总和的峰值，包含可观测到的 DataLoader workers |
| `cuda_peak_allocated_mib` | 本进程在选定 GPU 上 PyTorch allocator 的 allocated 显存峰值 |
| `cuda_peak_reserved_mib` | 同一 allocator 的 reserved 显存峰值，包含缓存池 |

CPU 使用 `psutil` 每 0.1 秒采样，并在阶段边界采样；它是**采样峰值**，可能遗漏短暂尖峰。
多进程 RSS 相加可能重复计算共享内存，不能解释为精确的独占物理内存。
CUDA 使用 PyTorch peak counters，不等于 `nvidia-smi` 所示全设备用量，不覆盖 TensorFlow、
其他进程或非 PyTorch 的 CUDA 分配。

每轮重置 CUDA 峰值计数器前先合并已有峰值，因此总 run 峰值不会因逐轮重置而丢失。
每次重复训练结束释放上一个服务器，下一次开始前清理未使用的 CUDA 缓存；
已加载的 Python/库内存仍可能保留，不保证多个 run 拥有完全相同的内存基线。
CPU 训练或调试时 CUDA 字段为 JSON `null` / CSV 空值 / NPZ `NaN`，**不是伪造为0**。
采样数量、间隔和权限/采样异常也记录在 JSON 中。

## 每轮 train loss

默认 `steps` 模式的 `train_loss` 是该轮所有实际本地训练 minibatch 的 **total objective 算术平均值**：

```text
train_loss(round) = sum(actual_total_loss_of_each_local_batch) / train_steps
```

CIFAR-10/SVHN 的 FedCyon、FedPDM、SAGE-Pure 可通过
`--local_training_mode epochs --local_epochs E` 切换为完整本地 epoch。该模式保留尾批，
所以 `train_loss` 和按客户端类型分组的 loss 都按**该批实际样本数**加权：

```text
train_loss(round) = sum(batch_objective * batch_sample_count) / train_samples
```

`train_samples` 是该轮所有参与客户端的样本访问次数，多个 epoch 的重复访问重复计数，
但同一样本的 weak/strong 两种增强只算一次。它不是去重聚合样本数。
JSON/CSV/NPZ 均新增此字段；steps 模式为缺失值，不用步数估算。
JSON 同时记录 `local_training_mode` 和 `local_epochs`；所有 loss 都是训练过程中的观测，
不是固定模型上完整数据集的离线 loss。不同客户端数据量不等时，不再让每个尾批拥有完整 batch 的权重。
各方法原有分组日志仍按客户端平均，应与统一 recorder 的曲线字段区分。

- 记录传入 backward 前、GradScaler 缩放前的目标，包含该方法实际使用的无标签系数和正则项。
- 不是验证集 CE，不用最后一个 batch 代替整轮均值，也不简单相加两类客户端的均值。
- 只统计本轮真正执行的本地训练批次；不按全体客户端数稀释。
  RSCFed 的多个子组中重复训练同一客户端，每次实际执行的批次都计入。
- `train_steps` 是目标观测次数；AMP 因非有限梯度跳过 optimizer step 时，它不是成功更新数。
- 辅助字段 `labeled_train_loss`、`unlabeled_train_loss`、`mixed_train_loss` 分别按纯有标签、
  纯无标签、混合客户端的**完整目标**分组平均；不是从混合客户端中拆出的监督/无监督分量。
- `participating_clients` 为本轮贡献训练批次的去重客户端数。
- 未执行任何训练步骤的轮次 loss 为缺失值，不填0；出现非有限 objective 时该轮 loss 也为缺失值，
  同时记录 `nonfinite_loss_steps`。0.0 仍是合法值，例如所有伪标签被拒绝时。
- 不保留 autograd 图，不存原图，不将每批预测/伪标签写盘。

同一方法不同实验可以直接用它看训练目标的收敛趋势；不同方法的目标定义和尺度可能不同，
不应只凭 loss 更低就判断方法更好。跨方法效果仍比较相同评估协议下的 Accuracy 等指标。

ImageNet-100 原 `run_<run>.json` 的 `training` 记录中也加入这些逐轮字段，
并用 `training_metrics_file` 指向完整的训练统计 JSON。原评估 NPZ 中的 `loss`
**仍是验证集 loss**，没有改名或偷换含义；画训练曲线请读取新的 `_training.npz`。
旧数据集原有 `.npy` 指标输出保持原样。

## 画 convergence curve

在项目根目录，用现有 matplotlib 依赖即可画图：

```python
import matplotlib.pyplot as plt
import numpy as np

with np.load("outputs/metrics/imagenet100/fedpdm_imagenet100_cmc_label01/run_1_training.npz",
             allow_pickle=False) as metrics:
    plt.plot(metrics["round"], metrics["train_loss"], label="FedPDM")
plt.xlabel("Communication round")
plt.ylabel("Mean local training objective")
plt.legend()
plt.tight_layout()
plt.savefig("convergence.png", dpi=200)
```

画训练时间轴时，将 `round` 换成 `cumulative_training_time_seconds`。
CSV 也可直接用 pandas 或绘图软件打开。多随机种子的均值/标准差仍需在这些独立 run 上汇总。

## 验证范围

- 2026-10-20 补齐适配后的指定 18 组合各完成 3 轮 CPU 核心链路检查；训练时间、CPU 峰值、逐轮 loss 与输出格式验证见 `ADAPTER_TEST_REPORT.md`。新增 RSCFed 将同轮所有子组的实际目标均计入该轮统计。

- 已检查统计均值/步数、零损失、空轮次、非有限目标、异常与中断保存，以及 JSON/CSV/NPZ 一致性。
- 已用桩客户端执行六种旧服务端的真实训练循环，覆盖 CBAFed 预训练、IsoFed 双阶段和 RSCFed 多子组计数。
- FedCyon、FedPDM、SAGE-Pure 已完成合成 ImageNet-100 数据的两轮 CPU 联调；
  FedPDM / SAGE-Pure 另用真实图像和 ResNet18 各完成两个客户端的两轮 CPU 小步训练，验证新增统计。
  小步测试把置信阈值设为0以覆盖无标签反向分支，不代表默认配置的收敛结果。
- CUDA 同步、峰值重置和重复 run 清理已做模拟 API 检查，尚未在真实 GPU / AMP 下验证。
  2026-10-20 的验证没有新增 CPU/MPS 训练入口；2026-10-21 已为 ImageNet-100 的 FedCyon / FedPDM 开放 `--device cpu --amp false`，沿用相同统计记录器。默认仍为 CUDA，MPS 未接入。

- 2026-10-21：项目 `.venv` 下通过正式 `train.py` 入口，各完成 1 轮真实 ImageNet-100 训练，保持默认 20 客户端、每客户端 5 步、batch size 16、4 CPU 线程；初始与最终各评估全部 5000 张验证图像。FedCyon / FedPDM 的训练阶段分别约 108.47 / 97.77 秒，总耗时约 270.98 / 283.10 秒，采样 RSS 峰值约 2925 / 3227 MiB。JSON / CSV / NPZ 的 loss、100 次目标观测、20 个参与客户端、CPU 设备和空 CUDA 指标均检查通过。该短测不代表 200 轮收敛或 CUDA/MPS 验证。原始记录：`outputs/smoke_tests/20261021_cpu_cli_142837/`（Git 忽略，仅保存在本机）。
