# 七个组合的适配与 18 组合回归

日期：2026-10-20。范围只包括 FixMatch、FedConsist、RSCFed、FedCyon、FedPDM、SAGE-Pure × CIFAR-10、SVHN、ImageNet-100。未扩展 CBAFed、IsoFed 或其他数据集。

## 1. 当前结论

**原来未接通的 7 个组合已补齐代码、参数检查、默认配置与结果记录；全部 18 个组合重新完成 3 轮小规模 CPU 核心链路测试。**

| 方法 | CIFAR-10 | SVHN | ImageNet-100 |
| --- | --- | --- | --- |
| FixMatch | 旧路径回归通过 | 旧路径回归通过 | 新适配通过 |
| FedConsist | 旧路径回归通过 | 旧路径回归通过 | 新适配通过 |
| RSCFed | 旧路径回归通过 | 旧路径回归通过 | 新适配通过 |
| FedCyon | 旧路径回归通过 | 旧路径回归通过 | 原索引路径回归通过 |
| FedPDM | 新适配通过 | 新适配通过 | 原索引路径回归通过 |
| SAGE-Pure | 新适配通过 | 新适配通过 | 原索引路径回归通过 |

“通过”不是正式 GPU 训练或论文结果复现：本机为 16 GiB Apple Silicon Mac，Python 3.12、torch 2.14.1、torchvision 0.29.1，使用 CPU、2 个计算线程、关闭 AMP。正式 `train.py` 仍要求 CUDA，18 个组合的原生入口均止于 CUDA 环境检查，未声称通过真实 CUDA/AMP 验证。

## 2. 改动范围与算法边界

### ImageNet-100：三个旧方法的纯客户端适配

- `User/user_imagenet_adapters.py`：FixMatch/FedConsist 有标签端使用基础视图 CE，无标签端复用现有 `consistency_loss`。FixMatch 使用配置的温度、阈值、标签类型与无标签系数；FedConsist 保留固定温度 1、阈值 0.5、系数 1。两者保留旧优化器的 SGD Nesterov 和 BN/bias 权重衰减分组。
- `Server/serverImageNet100Legacy.py`：FixMatch/FedConsist 按参与客户端等权平均，沿用旧实际执行路径；没有误接 FedCyon 的 TA 聚合，也没有把未调用的旧 helper 当作有效算法。
- RSCFed 客户端保留持久 EMA teacher，弱增强预测按温度 0.5 锐化，与强增强在线模型计算概率 MSE。EMA 参数衰减为 `min(1 - 1/(step + 1), 0.999)`；teacher 保持 train-mode BN，不复制在线 BN buffers，不因新一轮广播而重置。
- RSCFed 每个子组从该通信轮起始全局模型训练。参考模型由样本数加权并应用 `model_m`；距离为各 state tensor L2 范数之和，权重正比于 `n_i * exp(-dist_scale_f * distance_i / n_i)`。可再分配标签组/无标签组质量，最后等权平均子组。整数 buffers 在参考更新、组内聚合、组间聚合各阶段按目标类型转换。
- **明确的工程修正**：旧 RSCFed 的纯无标签客户端计数可能留为 0；新适配对两组都使用真实本地样本数，log-space 计算稳定权重，单组子组归一化为总质量 1。不声称与旧数值缺陷逐位一致。
- 新路径采用固定本地优化步数、恒定学习率和独立保留的客户端 optimizer，不运行旧全训练集梯度诊断；不新增 Mixed 客户端协议。FixMatch/FedConsist 中未参与实际损失或上传的旧辅助 EMA/诊断状态不迁入新路径。

### CIFAR-10 / SVHN：FedPDM 与 SAGE

- `Server/serverSmallImageBaselines.py`：复用 FedPDM/SAGE 既有目标、teacher 语义和聚合，不改成统一 FedAvg。CIFAR-10 使用项目 ResNet18，SVHN 使用 VGG19-BN，输入均为 32×32。
- `datautils/small_image_fssl.py`：复用共享图像库和客户端索引。默认四份配置显式复用对应的 20 客户端缓存；也可用 `--partition_file none` 生成基于 seed 的新索引。新划分不依赖缺失的旧聚类文件。
- 有标签客户端为前 `num_labeled_users` 个；无标签 loader 的 `targets=None`，不会读取其真实标签计算训练目标或先验。
- 新路径使用正确的 CIFAR-10 test / SVHN test split；旧公共 SVHN 评估使用 train 的历史行为未被静默修改，因此旧指标不能直接作为新路径的可比基线。
- FedPDM 仍上传 EMA 模型、按有标签样本数/本轮有效无标签样本数加权；SAGE-Pure 仍用冻结广播 teacher 的置信校正 KL、按样本数聚合在线模型。SAGE-Pure 仍是协议适配版，不是论文 Mixed 设定的完整复现。

### 共享代码

统一入口按方法和数据集选择服务端。公共索引路径提供可覆盖的数据、增强、模型构造接口；旧服务端默认评估行为不变。训练耗时、CPU/CUDA 内存、逐轮实际优化目标均使用现有 recorder，新路径输出按 `<dataset>/<save_name>/run_N*` 隔离。

## 3. 七份新配置

均位于 `config/experiments/`：

| 文件 | 入口 |
| --- | --- |
| `ImageNet100_FixMatch_label01.yaml` | `--method fixmatch` |
| `ImageNet100_FedConsist_label01.yaml` | `--method fedconsist` |
| `ImageNet100_RSCFed_label01.yaml` | `--method rscfed` |
| `CIFAR10_FedPDM_label01.yaml` | `--method fedpdm` |
| `CIFAR10_SAGE_label01.yaml` | `--method sage` |
| `SVHN_FedPDM_label01.yaml` | `--method fedpdm` |
| `SVHN_SAGE_label01.yaml` | `--method sage` |

ImageNet 默认 20 客户端、前 2 个有标签、beta 0.5、seed 2026、ResNet18，无预训练。RSCFed 默认 4 子组 × 每组 12 客户端，不能仅按通信轮数与其他方法等同计算量。小图默认复用 beta 0.1 的 20 客户端缓存；此时配置 seed 不重新作用于旧索引。所有配置都是工程起点，不是论文最优设置。

## 4. 实际测试设置

全部组合为 3 轮、2 客户端（1 有标签 + 1 无标签）、每客户端每次参与 1 步、batch size 2；RSCFed 每轮 2 子组，每组都选中这两个客户端。

- ImageNet-100 六组合：使用真实原图符号链接，每类 4 张 train、1 张 val，共 400/100 张；真实 torchvision ResNet18、224 像素增强；第 0、1、2、3 轮评估。生产算法、模型及设备转移代码没有测试替换。
- 小图新增四组合：使用真实训练缓存中原客户端 0、2 各 32 张；真实 ResNet18/VGG19-BN。只替换原始评估数据入口，用训练缓存中每类 2 张、共 20 张真实图像诊断；不替换算法、模型或设备转移代码。
- 小图原有八组合：复用上一轮 CPU harness，在独立子进程内将旧 CUDA 调用转为 CPU、硬编码分组缩成 1+1，并提供同样的真实缓存数据和诊断评估集。原磁盘算法源码不因这些 runtime adapters 被改写。详细边界参见历史 `SMOKE_TEST_REPORT.md`。

**小图 Accuracy 不是独立测试集精度。** 本机未成功下载官方完整 CIFAR-10/SVHN 评估数据；真实下载、全量评估、正式 CUDA/AMP 仍待服务器验证。新后端正确选择 test split 已通过模拟 torchvision 构造器参数检查，但该检查不是完整数据下载或解码测试。

## 5. 结果及额外检查

18 个组合均完成：3 轮、有限 train loss、两组客户端 optimizer 调用、全局模型更新且参数有限、训练耗时和 CPU 内存记录、JSON/CSV/NPZ 写盘。新索引路径还检查评估轮次 `[0, 1, 2, 3]`、聚合权重和为 1、无标签 targets 为空及客户端状态回到 CPU。

新增七组合的逐轮目标如下，只用于确认数值有效，不能跨方法比较收敛或精度：

| 数据集 | 方法 | Round 1 | Round 2 | Round 3 |
| --- | --- | ---: | ---: | ---: |
| ImageNet-100 | FixMatch | 2.338434 | 2.874904 | 3.062273 |
| ImageNet-100 | FedConsist | 2.338434 | 3.960965 | 3.082114 |
| ImageNet-100 | RSCFed | 2.205807 | 3.530208 | 2.936840 |
| CIFAR-10 | FedPDM | 1.266795 | 0.720971 | 0.934486 |
| CIFAR-10 | SAGE-Pure | 1.480770 | 0.774953 | 1.135840 |
| SVHN | FedPDM | 0.680976 | 0.437952 | 0.341141 |
| SVHN | SAGE-Pure | 0.957962 | 0.792060 | 0.131243 |

默认高阈值可能拒绝所有伪标签，optimizer 调用不等于有效无标签梯度。为此，四个小图新组合另以诊断覆盖 `p_cutoff=0` 各跑 3 轮，检查全部无标签样本通过、无标签 loss > 0 且聚合计数 > 0。该额外测试不改变默认配置，也不是默认设置的收敛结果。

数值与数据检查共 9 组：

1. 18 个组合参数解析、服务端导入；另执行 18 个实际 CLI dry-run。
2. 活动 Python 源码 AST、107 份 YAML 语法。
3. 非法迭代数、客户端数、非有限损失/优化器参数、未实现 resume、错误聚合设置的拒绝，以及显式缓存取消选项。
4. FixMatch/FedConsist 硬/软 CE、阈值掩码、detach 和梯度的数值对照。
5. RSCFed MSE、EMA 前两步、持久 teacher、仅参数 EMA、teacher BN 行为。
6. RSCFed 解析权重、0.7/0.3 组质量、单组归一化、极端距离尺度稳定性。
7. 浮点加权及逐阶段整数 buffer 转换；保留 FedPDM/SAGE 的最大权重客户端整数 buffer 策略。
8. 模拟 torchvision API 检查 CIFAR-10 `train=False` 和 SVHN `split='test'`。
9. 合成小图确定性分区、缓存复用、跨 seed 共用像素库、显式缓存来源记录、错误缓存拒绝。

另通过真实 ImageNet 符号链接执行了 `train.py --method rscfed ... --prepare-data`。运行日志中的 CPU pin-memory 和旧 API 等 warning 不代表训练 error；没有为消除 warning 修改旧算法。

## 6. 保存的证据与复跑

目录：`outputs/smoke_tests/20261020_seven_adapters/`。

- `matrix.json`：18 个组合及额外无标签分支测试的结果汇总。
- `<dataset>_<method>.json/.log`：10 个新索引路径组合的结果和完整日志。
- `legacy_regression/`：8 个旧小图组合的独立结果和日志。
- `unlabeled_branch/`：四个 `p_cutoff=0` 的额外诊断测试。
- `adapter_checks.json`：9 组数值/数据检查；`preflight/` 保存真实入口解析和 CUDA 缺失日志。
- `metrics/` 及子目录：原始训练时间、内存、逐轮 loss 与评估文件。
- `adapter_validation.py`、`adapter_checks.py`、`legacy_cpu_validation.py`、`adapter_preflight.py`：本次测试脚本，使用项目 Python 环境执行；它们是诊断 harness，不是正式训练入口。

例如重跑某个新增 CPU 核心链路（新建输出目录避免覆盖）：

```sh
python outputs/smoke_tests/20261020_seven_adapters/adapter_validation.py \
  --single rscfed imagenet100 --output outputs/smoke_tests/local_adapter_check
python outputs/smoke_tests/20261020_seven_adapters/adapter_validation.py \
  --single fedpdm svhn --exercise-unlabeled --output outputs/smoke_tests/local_unlabeled_check
```

正式 GPU 检查使用 `train.py`，不是这些 CPU harness。例如先准备小图的独立测试数据，再运行：

```sh
python train.py --method fedpdm --config config/experiments/CIFAR10_FedPDM_label01.yaml \
  --prepare-data --download_data true
python train.py --method fedpdm --config config/experiments/CIFAR10_FedPDM_label01.yaml \
  --gpu 0 --num_global_iters 3 --num_subset_users 2 --local_iterations 1 \
  --batch_size 2 --eval_interval 1 --save_name gpu_smoke_cifar10_fedpdm
python train.py --method rscfed --config config/experiments/ImageNet100_RSCFed_label01.yaml \
  --gpu 0 --num_global_iters 3 --num_subset_users 2 --local_iterations 1 \
  --batch_size 2 --S_M 2 --S_K 2 --S_K_lb 1 --eval_interval 1 \
  --save_name gpu_smoke_imagenet100_rscfed
```

这些命令保留正式配置的 20 个总客户端和完整测试/验证集。小图显式缓存必须随共享像素库一起复制；要减少总客户端数，需换成匹配的新索引，不能只覆盖 `num_total_users`。新旧 SVHN 的正式公平精度比较仍须先统一评估协议。
