# 指定六种方法 × 三个数据集：3 轮冒烟测试

> 历史报告：这是补齐适配**之前**的测试记录。2026-10-20 已补齐下述 7 个未适配组合，并重新验证全部 18 个组合；请以 `ADAPTER_TEST_REPORT.md` 查看当前适配状态、执行结果与限制。本文保留原始结论，不追溯修改为通过。

测试日期：2026-10-20。范围仅包含 FixMatch、FedConsist、RSCFed、FedCyon、FedPDM、SAGE-Pure，以及 CIFAR-10、SVHN、ImageNet-100；未测试其他方法或数据集。

## 1. 结论与边界

**18 个组合中，11 个已接入组合完成了 CPU 小规模核心链路的 3 轮训练，未出现训练异常；另外 7 个组合尚未适配，正式入口在参数检查阶段明确拒绝。**

**这不是 CUDA 正式训练通过的结论。** 当前机器是 16 GiB 内存的 Apple Silicon Mac，PyTorch 检测不到 CUDA。11 个已支持组合通过原始 `train.py` 启动时均因 CUDA 要求而停止，未进入训练。因此补充执行了下述 CPU 核心链路测试，不能将其替代服务器上的原配置、原数据准备、CUDA/AMP 验证。

本次没有修改项目算法源码、损失公式、聚合公式、模型结构或正式入口；测试适配仅发生在独立子进程内存中。

## 2. 组合矩阵

“CPU 3 轮通过”表示第 3 节限定条件下的真实优化和聚合，不表示默认 20 客户端的 CUDA 实验已经跑通。

| 方法 | CIFAR-10 | SVHN | ImageNet-100 |
| --- | --- | --- | --- |
| FixMatch | CPU 3 轮通过 | CPU 3 轮通过 | 未适配 |
| FedConsist | CPU 3 轮通过 | CPU 3 轮通过 | 未适配 |
| RSCFed | CPU 3 轮通过 | CPU 3 轮通过 | 未适配 |
| FedCyon | CPU 3 轮通过 | CPU 3 轮通过 | CPU 3 轮通过 |
| FedPDM | 未适配 | 未适配 | CPU 3 轮通过 |
| SAGE-Pure | 未适配 | 未适配 | CPU 3 轮通过 |

尚未适配的组合不是随机训练崩溃，而是当前接口的明确限制：

```text
FedPDM / SAGE + CIFAR-10 或 SVHN:
The new fedpdm/sage implementations currently support ImageNet-100 only

FixMatch / FedConsist / RSCFed + ImageNet-100:
ImageNet-100 supports --method fedcyon, fedpdm or sage; the other five server paths are unchanged
```

不能仅绕过配置检查就认为剩余组合已经完成适配。本次按测试请求保留这些限制，没有扩展算法支持范围。

## 3. 测试实际怎么运行

### 环境及公共设置

- Python 3.12，PyTorch 2.14.1，macOS arm64。
- 实际计算设备为 CPU；未使用 MPS，AMP 关闭。
- 每个通过的组合真实执行 3 个通信轮次。
- 2 个客户端：1 个纯有标签、1 个纯无标签，每轮均参与。
- 每客户端每次参与训练 1 个 minibatch，batch size 为 2；训练 seed 固定为 11。
- RSCFed 使用 2 个子组、每组 2 个客户端，因此每轮各客户端实际训练两次。
- 保留各方法所选 YAML 的学习率、损失配置、置信阈值等；没有为了通过测试统一把阈值改为 0。
- 复用项目实际模型，不替换成线性层或虚假网络。使用实际客户端训练、backward、optimizer step 和服务端聚合，未使用假客户端返回固定 loss。

### CIFAR-10 / SVHN：CPU 适配后的旧核心流程

原代码多处硬编码 `.cuda()` 和客户端分组。测试脚本仅在独立子进程中：

1. 将 CUDA tensor/model 转移调用映射为 CPU 转移，移除 CUDA-only 构造检查，跳过设备选择调用；没有伪造 `torch.cuda.is_available()` 的全局返回值。
2. 将服务端硬编码的客户端分组缩为 `[1, 0, 1]`，用于小规模检查。
3. 从已有真实共享图像缓存中读取原客户端 0、2，每个取 32 张图像，作为测试客户端 0、1。
4. 用缓存中的真实图像构造每类 2 张、共 20 张的诊断评估集；执行原评估函数，包括其原有梯度诊断。

数据来源：

```text
data/cifar10_20_noniid-labeldir_0.1.npz
data/svhn_20_noniid-labeldir_0.1.npz
```

**这两个数据集的评估样本来自训练缓存，不是官方独立测试集，因此测试输出中的 Accuracy/AUC 等不能作为论文结果或泛化性能报告。** 本机没有完整的 CIFAR-10/SVHN 原始评估目录；原始数据下载、新划分生成、FixMatch noise 后缀缓存匹配等并没有在此次 CPU 适配测试中得到验证。

上述适配没有修改磁盘上的 Server/User 源码，也没有改变原有共享图像库或实验索引。

### ImageNet-100：直接使用 CPU 设备的现有服务端

- 从现有真实原图中每类选 4 张训练图、1 张验证图，共 400 张训练图、100 张验证图。
- 测试目录保存指向原图的符号链接，没有复制图像像素。
- 使用项目原始 ImageNet-100 数据准备逻辑，在独立测试缓存中划分成 2 个客户端，保留 100 类。
- 使用标准 torchvision ResNet18、项目原有 224 像素增强和评估。
- 直接给现有服务端传入 `torch.device('cpu')`；没有替换该路径的模型、损失、聚合或服务端实现。仅未经过正式入口的 CUDA-only 检查。
- 在第 0、1、2、3 轮评估，验证集仍来自原始 `val` split，只是缩小为 100 张。

## 4. 检查内容及结果

每个通过组合都检查了：

1. `Server.process()` 正常返回，统计状态为 `completed`，完成轮数恰好为 3。
2. 每轮都有实际训练目标记录，`train_loss` 有限，没有 NaN/Inf。
3. optimizer 的 post-step hook 观测到有标签、无标签客户端都执行了更新调用。
4. 最终全局模型参数全部有限，抽查的首个参数张量相对初始状态确实改变。
5. `_training.json`、`_training.csv`、`_training.npz` 均生成，三轮字段/数值对齐。
6. CPU 内存峰值和训练时间记录存在；CUDA 指标正确为缺失值，而不是假报 0。
7. 原有评估结果写盘成功：旧数据集 Accuracy 数组长度为 3；ImageNet-100 评估轮数为 `[0, 1, 2, 3]`。

下表是本次小规模执行的逐轮优化目标，仅用于确认数值有效，不能跨方法比较精度或作为收敛实验。

| 数据集 | 方法 | 第 1 轮 loss | 第 2 轮 loss | 第 3 轮 loss | 每轮目标观测数 |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | FixMatch | 1.498430 | 0.583895 | 0.000497 | 2 |
| CIFAR-10 | FedConsist | 1.472011 | 0.685343 | 0.472540 | 2 |
| CIFAR-10 | RSCFed | 1.327174 | 4.432153 | 2.750959 | 4 |
| CIFAR-10 | FedCyon | 1.296471 | 2.981950 | 1.716945 | 2 |
| SVHN | FixMatch | 1.031701 | 0.378941 | 0.044636 | 2 |
| SVHN | FedConsist | 1.150845 | 1.133638 | 1.112509 | 2 |
| SVHN | RSCFed | 1.150954 | 1.121677 | 1.075223 | 4 |
| SVHN | FedCyon | 1.212180 | 0.339033 | 0.058625 | 2 |
| ImageNet-100 | FedCyon | 2.337983 | 2.870471 | 2.900674 | 2 |
| ImageNet-100 | FedPDM | 2.278398 | 2.145194 | 2.146046 | 2 |
| ImageNet-100 | SAGE-Pure | 2.339023 | 2.356592 | 3.825713 | 2 |

非 RSCFed 组合的有标签/无标签 optimizer 调用数各为 3；RSCFed 各为 6。高阈值下某些无标签目标可能为零，optimizer 调用不保证该客户端每一步都有非零有效梯度；没有把零伪标签通过率错误判为异常。

日志中有旧 AMP API 弃用、CPU 下禁用 CUDA GradScaler 及小样本分类指标等 warning；不影响本次完成。没有借测试修改这些算法/API 行为。

## 5. 测试产物与复跑

统一输出目录：

```text
outputs/smoke_tests/20261020_requested_matrix/
├── matrix.json                    18 个组合的机器可读总结果、环境、限制
├── <dataset>_<method>_cli.log      正式 CLI 的实际拒绝/设备错误
├── <dataset>_<method>_cpu.log      已适配组合的真实 CPU 核心执行日志
├── <dataset>_<method>_cpu.json     每个组合的完整结果和适配说明
├── cpu_smoke_harness.py           本次使用的测试脚本，不是正式训练入口
├── image_links/                   真实 ImageNet 图片的测试符号链接
├── cache/                         隔离的 ImageNet 测试划分
├── saved_models/                  ImageNet 测试数据摘要
├── metrics/                       原评估文件、时间/内存/逐轮 loss
└── harness_first_attempt/         CPU 测试适配器首次失败的审计记录
```

用已安装完整 Mac 依赖的 Python，在项目根目录复跑：

```sh
python outputs/smoke_tests/20261020_requested_matrix/cpu_smoke_harness.py \
  --output "$PWD/outputs/smoke_tests/requested_matrix_rerun"
```

应使用新的输出目录。脚本专为本机 CPU 诊断编写，包含显式内存适配，**不能把它的成功当作 CUDA 正式训练的成功**；输出目录被 Git 忽略，转移机器时需要单独保存。

审计说明：第一次 CIFAR-10/FedCyon CPU 检查仍命中了客户端的复合 CUDA 守卫，属于测试适配器遗漏，不是算法异常。补全测试适配后重新执行完整 3 轮并通过；首次记录单独保留，最终 `matrix.json` 标明了重试原因。生产代码未改动。

## 6. 仍待验证 / 需要适配的部分

- 11 个已支持组合：仍需在真实 CUDA 服务器上用正确原始数据进行正式入口 3 轮检查，包括默认客户端数量下的显存/内存和 AMP 行为。
- 旧数据的新划分生成、缺失聚类文件、正式评估 split、固定 20 客户端分组等限制仍以 `README.md` 为准；本次 CPU 小规模适配没有修复或证明它们可用。
- 7 个未支持组合：需要先扩展对应客户端、数据构造及服务端路径，不能仅凭当前 smoke 结果列为已有 baseline。
- 本次只检查短程执行和产物，不判断长期收敛、最终精度、跨 seed 稳定性或论文复现程度。
