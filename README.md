# Betelgeuse：联邦半监督图像分类实验

Betelgeuse 是一个面向图像分类的 **Federated Semi-Supervised Learning（FSSL，联邦半监督学习）** 项目。数据分散在多个客户端：部分客户端拥有标签，其余客户端依赖伪标签和一致性学习；服务端按通信轮次下发模型、组织本地训练、聚合更新并评估。

当前以 **FedCyon** 为主要研究方法，保留六条旧流程，并增加 **FedPDM** 和 **SAGE-Pure（适配版）**。FixMatch、FedConsist、RSCFed、FedCyon、FedPDM、SAGE-Pure 已接通 CIFAR-10、SVHN、ImageNet-100 三个数据集，共 18 个组合。

> **先看运行边界**
>
> - 正式训练入口目前要求 NVIDIA CUDA；Mac 环境用于配置检查、数据准备和代码级 CPU 调试，不能直接通过 `train.py` 在 CPU/MPS 上训练。
> - ImageNet-100 已接入上述六种方法；ImageNet-1K 全量训练与 DomainNet 尚未接入。
> - 新路径已做 CPU 小规模联调，不等于完成真实 GPU/AMP 验证或复现论文精度。
> - 旧配置、旧评估和部分数据准备代码仍有历史限制。第 7、11 节列出了具体问题，不应把“有配置”理解为“所有组合都已完整测试”。

## 目录

1. 项目结构与代码阅读路线
2. 方法与数据集范围
3. 安装环境
4. 数据准备与划分
5. 入口、配置与常用参数
6. ImageNet-100：六个方法的实验步骤
7. CIFAR-10 / SVHN 与旧数据集：配置选择
8. 多次运行、后台日志与对照实验
9. 结果、耗时、内存与收敛曲线
10. 备份与磁盘占用
11. 常见问题与已知限制
12. 验证范围与文档导航

## 1. 项目结构与代码阅读路线

### 1.1 目录结构

```text
betelgeuse/
├── train.py                         统一入口，推荐从这里启动
├── main_*.py                        六种旧方法的兼容入口
├── experiments/
│   ├── config.py                    参数解析、YAML 合并、方法注册与合法性检查
│   ├── runner.py                    设备检查、随机种子、重复运行、服务器调度
│   ├── paths.py                     项目相对路径、数据及结果目录
│   └── recording.py                 耗时、内存、逐轮训练 loss 的统一记录
├── Server/
│   ├── serverbase.py                公共服务端接口及旧评估流程
│   ├── serverBalanced.py            旧数据集上的 FedCyon，类名 Balanced
│   ├── serverFixMatch.py            联邦 FixMatch 流程
│   ├── serverFedConsist.py          FedConsist 流程
│   ├── serverCBAFed.py              CBAFed，包括初始有标签预训练
│   ├── serverRSCFed.py              RSCFed，包含子组训练和聚合
│   ├── serverIsoFed.py              IsoFed，两阶段训练和更新
│   ├── serverImageNet100.py         ImageNet-100 FedCyon、共享评估与数据初始化
│   ├── serverImageNet100Baselines.py  FedPDM / SAGE 的共享调度与加权聚合
│   ├── serverImageNet100Legacy.py   ImageNet-100 FixMatch / FedConsist / RSCFed
│   └── serverSmallImageBaselines.py  CIFAR-10 / SVHN FedPDM / SAGE
├── User/
│   ├── userbase.py                  公共客户端状态与记录接口
│   ├── userreweight_mycb.py         FedCyon 本地目标、类别重加权和 TA
│   ├── userfixmatch.py              FixMatch 客户端，IsoFed 也复用
│   ├── userfedconsist.py            FedConsist 客户端
│   ├── usercbafed.py                CBAFed 客户端
│   ├── userrscfed.py                RSCFed 客户端
│   ├── user_fssl_baselines.py       FedPDMClient、SAGEClient 及核心损失
│   └── user_imagenet_adapters.py    FixMatch / FedConsist / RSCFed 的纯客户端适配
├── Models/                         旧路径的 ResNet、LeNet、VGG 等模型
├── datautils/
│   ├── imagenet100.py              类别清单、路径索引、客户端划分、224 像素增强
│   ├── small_image_fssl.py         小图新基线：共享索引、32 像素增强、官方 test split
│   ├── partition_store.py          旧数据集共享图像库与实验索引
│   ├── migrate_partitions.py       旧缓存迁移、验证、导出工具
│   ├── datasets.py                 旧数据集封装
│   ├── utils.py                    旧数据读取与划分
│   └── data_utils/                 DataLoader、采样器、数据增强与 SSL 数据包装
├── utils/                          优化器、损失、EMA、日志等公共工具
├── config/
│   ├── experiments/               107 份实验 YAML，含 6 份 ImageNet-100 配置
│   ├── datasets/imagenet100_cmc.txt 有序的 100 类 synset 清单
│   └── feature_partitions/         历史特征划分辅助文件，部分所需文件缺失
├── scripts/run.sh                 可从其他工作目录启动的包装脚本
├── data/
│   ├── datasource/                原始数据，默认每个数据集一个子目录
│   ├── _shared/                   共享图像库或 ImageNet-100 文件清单
│   ├── partitions/imagenet100/    ImageNet-100 客户端索引
│   ├── *.npz                     旧数据集客户端索引与标签
│   └── downloads/                保留的历史下载包
├── outputs/
│   ├── saved_models/             实验配置快照；目录名不代表自动保存了权重
│   ├── metrics/                  新实验指标
│   ├── logs/                     数据划分日志及手动重定向的训练日志
│   ├── legacy_results/           保留的历史实验结果
│   └── legacy_logs/              保留的历史日志
├── archive/                      历史入口、脚本、草稿与压缩源码备份
├── requirements.txt              两个平台共用的直接依赖
├── requirements-mac.txt          Mac 调试环境
├── requirements-cuda.txt         Linux CUDA 训练环境
├── DATA_STORAGE.md               图像去重、数据迁移和校验
├── BASELINES.md                  两个新 baseline 的实现及论文差异
└── TRAINING_METRICS.md            时间、内存、train loss 的精确定义
```

`outputs/metrics`、`outputs/saved_models` 等目录按需创建。保留大写的 `Server/`、`User/`、`Models/` 是为了兼容已有导入；`archive/` 中的脚本不作为推荐入口。

### 1.2 一次实验的执行链

```text
train.py
  → experiments.config.parse_args()
  → experiments.runner.run()
      → 解析配置、检查 CUDA、保存 resolved_config.json
      → 为每个 run 设置训练 seed，创建 TrainingRecorder
      → 构建对应 Server、加载或生成数据索引、创建 User
      → Server.process()
          → 评估 / 选择客户端
          → User.train()：前向、目标函数、反向与更新
          → Server 聚合
          → 保存逐轮训练统计及算法原有指标
      → 保存该 run 的时间和内存汇总，释放服务器
```

评估位置因路径不同：ImageNet-100 及 CIFAR-10/SVHN 的 FedPDM、SAGE 在初始状态和指定完成轮次后评估；旧流程通常在每轮本地训练前评估。详细区别见第 9 节。

### 1.3 想改什么，就先看哪里

| 任务 | 首先阅读 |
| --- | --- |
| 改 FedCyon 的 user objective | `User/userreweight_mycb.py` |
| 改 FedCyon 的 server aggregation | 旧数据用 `Server/serverBalanced.py`；ImageNet-100 用 `Server/serverImageNet100.py` |
| 改 FedPDM | `User/user_fssl_baselines.py` 中 `FedPDMClient`、`match_class_distribution()` |
| 改 SAGE-Pure | 同文件中的 `SAGEClient`、`sage_consistency_loss()` |
| 改两个新方法的聚合 | `Server/serverImageNet100Baselines.py`，以及客户端的 `aggregation_count()` / `upload_state()` |
| 改数据划分 | ImageNet-100 看 `datautils/imagenet100.py`；小图新基线看 `datautils/small_image_fssl.py`；旧流程看各 Server 的 `prepare_data()` 和 `datautils/utils.py` |
| 改保存内容 | `experiments/recording.py` 和各 Server 的 `new_result_save()` |
| 增加 CLI 参数 / 方法入口 | `experiments/config.py`、`experiments/runner.py` |

## 2. 方法与数据集范围

### 2.1 方法入口

| 方法 | `--method` | 服务端 | 客户端 |
| --- | --- | --- | --- |
| FedCyon | `fedcyon` | `Balanced` / `ImageNet100FedCyon` | `UserReweight_mycb` |
| 联邦 FixMatch | `fixmatch` | `FixMatch` / `ImageNet100Legacy` | `UserFixMatch` / `FixMatchClient` |
| FedConsist | `fedconsist` | `FedConsist` / `ImageNet100Legacy` | `UserFedConsist` / `FedConsistClient` |
| CBAFed | `cbafed` | `CBAFed` | `UserCBAFed` |
| RSCFed | `rscfed` | `RSCFed` / `ImageNet100Legacy` | `UserRSCFed` / `RSCFedClient` |
| IsoFed | `isofed` | `IsoFed` | `UserFixMatch` |
| FedPDM | `fedpdm` | `ImageNet100Baseline` / `SmallImageBaseline` | `FedPDMClient` |
| SAGE-Pure（adapted） | `sage` | `ImageNet100Baseline` / `SmallImageBaseline` | `SAGEClient` |

**`--method` 选服务端流程；`name_algorithm` 选客户端数据视图/损失相关分支；`aggregate_method_server` 不是统一的方法选择器。** 不要只改后两个参数来切换方法。

FedPDM 是独立重实现，仍采用项目的固定本地步数等工程设置。SAGE-Pure 适配了“纯有标签 / 纯无标签客户端”协议，不是原论文 Mixed 客户端设定的完整复现。方法参数、公式与差异记录在 `BASELINES.md`。

### 2.2 数据集范围

| 数据集 | `--dataset` | 类别数 | 当前范围 |
| --- | --- | --- | --- |
| CIFAR-10 | `cifar10` | 10 | 旧路径；新增 FedPDM/SAGE 共享索引路径，指定六方法已接通 |
| CIFAR-100 | `cifar100` | 100 | 旧路径，有历史配置；不能据此保证六种方法全部组合已验证 |
| SVHN | `svhn` | 10 | 指定六方法已接通；新增 FedPDM/SAGE 使用 test，旧路径评估 split 问题见第 11 节 |
| STL-10 | `stl10` | 10 | 旧路径，常用特征划分，有外部聚类索引依赖 |
| TinyImageNet | `tinyimagenet` | 200 | 有配置和加载分支，但旧评估存在阻塞问题，见第 11 节 |
| ImageNet-100 | `imagenet100` | 100 | FedCyon、FixMatch、FedConsist、RSCFed、FedPDM、SAGE-Pure |

代码中零散的 MNIST、FashionMNIST、CelebA、FEMNIST 等辅助加载器，**不代表已经接入并验证了完整实验流程**。这里不把它们列为现成实验。ImageNet-1K 原图目录可作为选取 100 类的来源，不等于支持 1000 类训练；DomainNet 尚未适配。

## 3. 安装环境

以下命令除另有说明外，均在项目根目录执行。两套文件按仓库当前固定版本安装，不宣称这些版本是“最新版本”，也不要混装在同一个虚拟环境。

### 3.1 Mac：配置和数据调试

目标：Python 3.12、Apple Silicon / arm64；依赖文件标注 macOS 14+。Intel Mac 不在这份环境的目标范围。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac.txt
python -m pip check
python -c "import torch, torchvision, psutil; print('torch:', torch.__version__); print('vision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml --dry-run
```

该文件固定 PyTorch `2.14.1`、torchvision `0.29.1`、TensorFlow `2.21.0`。TensorFlow 服务于旧数据加载，不安装 `tensorflow-metal`。即使检测到 MPS，项目也没有接通正式 MPS 训练；`--gpu -1` 不会切换为 CPU。

不要把临时 `/tmp` 调试环境当作项目的长期运行环境。

### 3.2 Linux：CUDA 正式训练

目标：Python 3.12、Linux x86_64；依赖文件标注 glibc 2.28+，使用 CUDA 12.6 的 PyTorch wheel。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -m pip check
nvidia-smi
python -c "import torch, torchvision, psutil; print('torch:', torch.__version__); print('vision:', torchvision.__version__); print('wheel CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda').item())"
```

该文件固定 `torch==2.14.1+cu126`、`torchvision==0.29.1+cu126`，TensorFlow 使用 `tensorflow-cpu`。需要可用的 NVIDIA 驱动和兼容 GPU；安装依赖成功不等于 CUDA 可用，以上实际张量运算检查不能省略。

`requirements.txt` 是公共依赖清单，通过两个平台文件的 `-r` 自动引入，包括 NumPy、Pillow、PyYAML、scikit-learn、matplotlib、psutil 等。**单独安装它不会安装 PyTorch / TensorFlow**；当前也没有完整锁定所有传递依赖。

### 3.3 不训练时的最小需求

- `python train.py --help`：不需要导入训练框架。
- `--dry-run`：读取 YAML 时需要 PyYAML，不导入 PyTorch，不进行数据存在性或 GPU 检查。
- ImageNet-100 `--prepare-data`：需要 NumPy、Pillow、PyYAML，不需要 CUDA，也不训练模型。
- 旧共享缓存的迁移/验证：NumPy 和标准库；使用 `fcntl`，面向 macOS/Linux。

## 4. 数据准备与划分

### 4.1 三种路径不要混淆

| 设置 | 默认值 | 含义 |
| --- | --- | --- |
| `--source_dir` / `BETELGEUSE_SOURCE_DIR` | `data/datasource` | 原始数据总目录，最终追加数据集名称 |
| `--datadir` | `<source_dir>/<dataset>` | 当前实验原始数据目录，显式设置时优先 |
| `BETELGEUSE_DATA_DIR` | 项目下的 `data` | 共享图像/清单和客户端划分缓存根目录 |
| `BETELGEUSE_CLUSTER_DIR` | `config/feature_partitions` | 旧特征聚类辅助文件目录 |

CLI 中的相对数据和输出路径以项目根目录为基准。环境变量建议使用绝对路径，尤其是 `BETELGEUSE_DATA_DIR`。

```sh
export BETELGEUSE_SOURCE_DIR=/datasets
export BETELGEUSE_DATA_DIR=/datasets/betelgeuse-cache
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml --datadir /datasets/imagenet100 --dry-run
```

`--data_dir` 是历史参数，**不是当前共享划分缓存根目录的替代开关**。仅改变 `save_name` 不会生成新的数据划分。

### 4.2 ImageNet-100 原图

默认目录布局：

```text
data/datasource/imagenet100/
├── train/
│   ├── nXXXXXXXX/*.JPEG
│   └── ...
├── val/
│   ├── nXXXXXXXX/*.JPEG
│   └── ...
└── source.json
```

`nXXXXXXXX` 是示意，实际使用 `config/datasets/imagenet100_cmc.txt` 中的 synset。类别文件必须恰好包含 100 个不重复类别，文件顺序决定标签编号。每个选中类别都需要训练和验证图像；额外类别不会加入训练。

本地工作副本已有社区镜像数据，来源记录见 `data/datasource/imagenet100/source.json`：126,689 张训练图、5,000 张验证图，原图文件总计约 8.47 GB（十进制）。该记录注明不是官方直接下载，未确认与官方原图逐字节相同。**数据目录被 Git 忽略，换机器或拿到源码副本后不能假设数据也在。** 不会通过 `--prepare-data` 自动下载 ImageNet。

若验证图像平铺在 `val/`，准备两列文本映射，每行是相对 `val/` 的路径和 synset，例如：

```text
ILSVRC2012_val_00000001.JPEG n02869837
```

这只是格式示例，不是该图片的真实标签。不能直接填 ILSVRC 数字类别编号；必须提供正确映射，并覆盖所选 100 类。

```sh
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml \
  --datadir /datasets/imagenet100 --imagenet_val_manifest /datasets/imagenet_val_synsets.txt --prepare-data
```

### 4.3 ImageNet-100 划分与共享存储

```sh
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml --prepare-data
```

该命令扫描文件、建立/复用清单和客户端索引，打印 JSON 摘要，不启动训练。摘要包括 `client_sizes`、`labeled_class_counts`、`classes_without_labels`、`manifest` 和 `partition_file`。扫描不会逐张解码所有图片，不能替代完整图像损坏检查。

六份默认配置采用相同数据协议：

| 设置 | 默认 |
| --- | --- |
| 客户端数 / 每轮参与数 | 20 / 20；RSCFed 为每轮 4 个子组、每组 12 个客户端 |
| 有标签客户端 | 前 2 个，其全部本地样本有标签 |
| 无标签客户端 | 后 18 个 |
| 划分 | `noniid-labeldir`，`beta=0.5` |
| 划分 seed | `partition_seed=2026` |
| 最少每客户端样本数 | `min_client_samples=10` |

`num_labels` 不限制这一路径的标注样本总数；2/20 是客户端比例，未必等于恰好 10% 的样本。可用 `--partition homo` 或调整 `--beta`、`--num_labeled_users`。除 RSCFed 外，客户端选择保证两组都参与，实际参与数等于 `num_subset_users`。RSCFed 使用 `S_M`、`S_K`、`S_K_lb` 控制子组，子组间可重复选中客户端，不能只用 `num_subset_users` 衡量其计算量。

缓存结构：

```text
data/_shared/imagenet100/<manifest_sha256>.json
data/partitions/imagenet100/<partition_metadata_sha256>.npz
```

原图只存一份；共享清单保存路径和标签，每个实验保存客户端索引。六种方法在相同数据参数下复用相同缓存。训练 seed 与划分 seed 独立；多次训练不自动改变划分。缺少标注覆盖的类别会报告，不会自动补充真实标签。

### 4.4 旧数据集

旧服务端先创建评估集，多处使用 `download=False`，因此即使已有训练缓存，也需要提前准备原始评估数据。默认读取位置为 `data/datasource/<dataset>/`。可在已安装环境中按需下载，例如 CIFAR-10：

```sh
python - <<'PYCODE'
from torchvision.datasets import CIFAR10
for training in (True, False):
    CIFAR10(root="data/datasource/cifar10", train=training, download=True)
PYCODE
```

CIFAR-100、SVHN、STL-10 应准备相应训练、测试及需要的无标签 split。TinyImageNet 的验证目录必须符合 `ImageFolder` 的类别目录要求，官方平铺 `val/images` 不能原样当作分类目录使用；此外仍有第 11 节的评估代码问题。

旧缓存由 `data/*.npz` 与 `data/_shared/<dataset>/` 共同组成；`.npz` 只有客户端图像编号、标签和校验信息，像素通过共享库加载。已有划分可避免重新生成，但不能替代原始评估数据。

**新建旧划分的限制：** `load_cifar10_data()` 和 `load_stl10_data()` 会读取以下缺失的聚类文件，其中 CIFAR-10 即使选择标签划分也会经过该读取逻辑：

```text
cifar10_match_client_index_cluster_20_24784754.txt
stl10_match_client_index_cluster_10_247890221.txt
```

应补齐正确的原文件或使用经过验证的已有缓存，不能把别的聚类文件改名冒充。FixMatch 的缓存名还额外包含 `noise` 后缀，所以其他方法已有同一 beta 的缓存，不代表 FixMatch 会命中同一文件。详见第 11 节。

### 4.5 CIFAR-10 / SVHN 的 FedPDM 与 SAGE

四份新配置默认通过 `partition_file: data/<dataset>_20_noniid-labeldir_0.1.npz` 复用已有共享划分，不重新分配像素。显式缓存必须与数据集和客户端数匹配；其中前 2 个客户端有标签，其余无标签。此时 `partition_seed`、`partition`、`beta` 不会重新切分缓存，数据摘要明确记录 `partition_parameters_applied=false`。

已有训练缓存不能代替原始测试集。若原始数据尚未准备，可显式允许 torchvision 下载（默认不联网下载）：

```sh
python train.py --method fedpdm --config config/experiments/CIFAR10_FedPDM_label01.yaml \
  --prepare-data --download_data true
python train.py --method sage --config config/experiments/SVHN_SAGE_label01.yaml \
  --prepare-data --download_data true
```

不使用旧划分时，加 `--partition_file none`，按 `partition_seed`、`partition`、`beta` 和客户端数从原始训练集生成确定性索引；不会读取缺失的旧聚类文件。不同划分继续共用 `data/_shared/<dataset>/` 的图像库。准备阶段需要 PyTorch/torchvision，不需要 CUDA。下载速度及上游可用性不属于本次冒烟测试保证范围。

新 CIFAR-10 路径使用 32×32 的项目 ResNet18，新 SVHN 路径使用 VGG19-BN；不把小图放大为 ImageNet 输入。评估分别使用 CIFAR-10 test 和 SVHN `split='test'`，不沿用旧 SVHN 的 train-split 评估问题。正式对照前仍需统一新旧流程的实际划分及评估协议。

## 5. 入口、配置与常用参数

### 5.1 先 dry-run，再训练

```sh
python train.py --help
python train.py --method fedcyon --config config/experiments/CIFAR10_Fedcyon_label01.yaml --gpu 0 --dry-run
```

优先级为 **命令行覆盖 YAML，YAML 覆盖方法默认值**。`--method` 默认 `fedcyon`，但推荐每次显式指定；不会根据配置文件名猜测算法。`method`、`c`、`dry_run`、`prepare_data` 必须放在命令行，不能作为 YAML 键。

旧 YAML 的 `sn` 映射到 `save_name`，`store_true` / `store_false` 和空加载路径会归一化。原 `learning_config/...` 路径也会兼容解析到新的配置目录。

```sh
python main_fedcyon.py --c learning_config/CIFAR10_Fedcyon_label01.yaml --dry-run
```

六个 `main_*.py` 分别固定方法。FedPDM/SAGE 使用统一入口，不存在新建的 `main_fedpdm.py` 或 `main_sage.py`。

从其他工作目录运行时可使用：

```sh
PYTHON=/path/to/venv/bin/python /path/to/betelgeuse/scripts/run.sh --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml --dry-run
```

### 5.2 常用参数

| 参数 | 用途 / 注意事项 |
| --- | --- |
| `--gpu 0` | 当前可见设备中的 CUDA 编号；不是 CPU 开关 |
| `--num_global_iters` | 通信轮数；ImageNet 配置为 200，很多旧配置为 2000 |
| `--local_iterations` | 本地更新的步数预算，不等价于遍历一遍本地数据 |
| `--local_epochs` | 旧算法使用方式各异；ImageNet 新基线要求为 1 |
| `--num_total_users` / `--num_subset_users` | 总客户端数 / 参与数；旧流程有硬编码限制 |
| `--batch_size` / `--uratio` | 基础 batch 大小 / 无标签 batch 比例 |
| `--eval_batch_size` | 评估 batch 大小 |
| `--eval_interval` | ImageNet-100 及 CIFAR-10/SVHN 新基线的评估间隔 |
| `--num_workers` | DataLoader 子进程数；调试先用 0 |
| `--lr` / `--weight_decay` | 学习率 / 权重衰减 |
| `--p_cutoff` / `--T` | 伪标签置信阈值 / 温度 |
| `--amp true` | 开启相关 AMP 分支，仍需在真实 GPU 上验证 |
| `--runtimes` / `--seed_offset` | 重复次数 / 训练种子偏移 |
| `--save_name` | 实验名称，隔离配置目录及新的逐轮统计 |
| `--metrics_dir` | 指标根目录，旧方法尤其需要独立设置以避免覆盖 |
| `--overwrite false` | 若配置保存目录已存在则拒绝开始 |

以下参数主要服务 ImageNet-100：`partition_seed`、`num_labeled_users`、`min_client_samples`、`imagenet_classes`、`imagenet_val_manifest`、`pretrained`、`ta_threshold`；FedPDM/SAGE 专有参数见 `BASELINES.md`。不要假设旧方法会响应相同的标注客户端设置或网络名称。

### 5.3 防止覆盖

默认 `overwrite=false`，但很多旧 YAML 写的是 `overwrite: store_true`。建议命令中显式写 `--overwrite false`，并为每次不同实验设置新名称。

`--overwrite true` 允许写入同名输出，**不是恢复训练**，也不会先递归删除目录。新 run 统计按 run 编号区分；旧 `.npy` 评估指标往往不包含 `save_name` 或 run 编号，因此仅改实验名称不能完全避免覆盖。

## 6. ImageNet-100：六个方法的实验步骤

这是目前新增 baseline 对照的共同数据路径。以下示例假定原图已在默认目录，若不在，每个命令都添加同一个 `--datadir /datasets/imagenet100`。

### 6.1 第一步：检查配置和数据

```sh
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml --dry-run
python train.py --method fedpdm --config config/experiments/ImageNet100_FedPDM_label01.yaml --dry-run
python train.py --method sage --config config/experiments/ImageNet100_SAGE_label01.yaml --dry-run
python train.py --method fixmatch --config config/experiments/ImageNet100_FixMatch_label01.yaml --dry-run
python train.py --method fedconsist --config config/experiments/ImageNet100_FedConsist_label01.yaml --dry-run
python train.py --method rscfed --config config/experiments/ImageNet100_RSCFed_label01.yaml --dry-run
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml --prepare-data
```

核对六种方法的客户端数、划分 seed、beta、类别文件和原图目录相同，再检查标注类别覆盖。`--prepare-data` 在 Mac 上也能执行。

### 6.2 第二步：GPU 小步检查

下面各运行两轮、每轮两个客户端、每客户端一步，验证真实 GPU 下的链路。仍使用完整验证集评估，因此不是“只读几张图”的测试。CPU 仍会创建 20 个客户端的模型状态。

```sh
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml \
  --gpu 0 --num_global_iters 2 --num_subset_users 2 --local_iterations 1 \
  --batch_size 2 --eval_batch_size 16 --eval_interval 1 --num_workers 0 \
  --save_name smoke_fedcyon_imagenet100 --overwrite false

python train.py --method fedpdm --config config/experiments/ImageNet100_FedPDM_label01.yaml \
  --gpu 0 --num_global_iters 2 --num_subset_users 2 --local_iterations 1 \
  --batch_size 2 --eval_batch_size 16 --eval_interval 1 --num_workers 0 \
  --save_name smoke_fedpdm_imagenet100 --overwrite false

python train.py --method sage --config config/experiments/ImageNet100_SAGE_label01.yaml \
  --gpu 0 --num_global_iters 2 --num_subset_users 2 --local_iterations 1 \
  --batch_size 2 --eval_batch_size 16 --eval_interval 1 --num_workers 0 \
  --save_name smoke_sage_imagenet100 --overwrite false
```

默认高阈值下，无标签分支可能没有通过筛选的样本。仅为调试反向分支，可以在独立 smoke 实验中添加 `--p_cutoff 0`；这会改变训练设定，不能当作默认阈值的收敛结果。

### 6.3 第三步：完整配置实验

```sh
python train.py --method fedcyon --config config/experiments/ImageNet100_Fedcyon_label01.yaml \
  --gpu 0 --save_name fedcyon_imagenet100_runA --overwrite false

python train.py --method fedpdm --config config/experiments/ImageNet100_FedPDM_label01.yaml \
  --gpu 0 --save_name fedpdm_imagenet100_runA --overwrite false

python train.py --method sage --config config/experiments/ImageNet100_SAGE_label01.yaml \
  --gpu 0 --save_name sage_imagenet100_runA --overwrite false
```

这些命令按顺序运行，不自动并发。默认预算为 200 轮、每轮 20 客户端、每客户端 5 步、batch 16，ResNet18 从零训练。方法默认差异包括：

| 设置 | FedCyon | FedPDM | SAGE-Pure |
| --- | --- | --- | --- |
| `name_algorithm` | `fixmatch` | `fedpdm` | `sage` |
| `p_cutoff` | 0.55 | 0.95 | 0.95 |
| `hard_label` | false | true | false |
| 上传与聚合 | 组内平均、组间 TA 权重 | EMA 模型、有效样本数权重 | 在线模型、样本数 FedAvg |

FedCyon 的 TA 是达到置信阈值的比例，不是真实准确率；两个新方法不使用其 TA 聚合。FedPDM 的无标签权重只统计本轮访问且最新伪标签通过阈值的不重复样本，不外推为全数据集的有效样本数，因此本地只训练少量步骤时可能偏重有标签端。

### 6.4 模型、资源和协议修改

- `--name_net resnet50` 可替换 ResNet18，但会增加资源需求；旧路径并不统一按该参数构造网络。
- 默认 `--pretrained false`。开启后使用 ImageNet-1K 监督预训练，属于额外监督，不能和从零训练实验混在同一设定中。
- 图像按需读取并裁剪到 224；客户端模型/优化器非活动时放在 CPU，只有当前客户端进入 GPU，聚合采用 CPU 累加。
- GPU 显存不足先减小 `batch_size`、`eval_batch_size`；CPU 内存仍需容纳多个客户端模型、优化器和 FedPDM 概率缓存。
- 改标注客户端数、beta、划分 seed 时，六种方法应使用相同设置；`--prepare-data` 会按新元信息生成或复用对应缓存。
- 评估集不能反复用来选择超参数后再当作独立最终测试集；正式调参需预先确定独立验证协议。

### 6.5 FixMatch / FedConsist / RSCFed 新适配

```sh
python train.py --method fixmatch --config config/experiments/ImageNet100_FixMatch_label01.yaml --gpu 0
python train.py --method fedconsist --config config/experiments/ImageNet100_FedConsist_label01.yaml --gpu 0
python train.py --method rscfed --config config/experiments/ImageNet100_RSCFed_label01.yaml --gpu 0
```

- FixMatch：基础视图有标签 CE；弱→强伪标签 CE，默认硬标签、阈值 0.95；按参与客户端等权聚合。
- FedConsist：保留旧纯客户端一致性分支，固定 `T=1`、阈值 0.5、无标签损失系数 1，默认软标签；按旧代码实际调用的等权路径聚合，而非未调用的重加权辅助函数。
- RSCFed：保留跨参与轮次的客户端 EMA teacher、温度 0.5 的概率锐化、概率 MSE；组内样本数/模型距离加权，默认标签组权重 0.7，组间等权。默认每轮 4 组、每组 12 客户端，不等于一次普通 FedAvg。

**RSCFed 的显式修正：** 新适配使用两组客户端的实际正样本数，避免旧无标签样本计数为 0 导致的退化权重；使用 log-space 稳定计算及单组归一化。因此是保留核心设计的工程适配，不是旧实现逐位复刻。教师的 train-mode BN、EMA 状态持久性和逐层整数 buffer 转换均已单独检查。

RSCFed 的 3 轮小步 GPU 检查可在上面命令后加：
`--num_global_iters 3 --num_subset_users 2 --local_iterations 1 --batch_size 2 --S_M 2 --S_K 2 --S_K_lb 1 --eval_interval 1 --save_name smoke_rscfed_imagenet100`。
这仍创建配置中的 20 个客户端并评估完整验证集；若同时改 `num_total_users`，须同步调整标注数和子组数量，并使用新数据划分。

## 7. CIFAR-10 / SVHN 与旧数据集：配置选择

### 7.1 配置命名不是完整实验语义

当前 `config/experiments/` 有 107 份 YAML，并不是所有“数据集 × 方法 × beta”的笛卡尔积。

- 旧 `FedAvg` 文件通常通过 `--method fixmatch` 运行联邦 FixMatch，不是新增的纯监督 FedAvg 方法。
- 旧 `label001 / label01 / label1 / label10` 在相应标签划分配置中通常代表 `beta=0.01 / 0.1 / 1 / 10`，**不是标注率**。
- ImageNet 六份配置虽叫 `label01`，实际 `beta=0.5`、2/20 客户端有标签。始终以解析后的字段为准。
- STL-10 主要使用 `feature...` 文件名，不存在 `STL10_Fedcyon_label01.yaml`。
- `*_Client20.yaml` 中有历史复制残留，例如 `CIFAR100_FixMatch_Client20.yaml`、`STL10_FixMatch_Client20.yaml` 的 `dataset` 实际都是 `cifar10`；部分聚合字段也有拼写/兼容性问题。不要只按文件名批量运行。

### 7.2 CIFAR-10 上启动六种旧方法

**前提：** CUDA 环境、原始评估数据、对应缓存或所需外部划分文件已准备好。下面给出当前入口与历史配置的正确搭配；不是六种方法在当前依赖上均已完成全量训练的保证。

FedConsist、CBAFed、RSCFed、IsoFed 复用 `FedAvg` / `Fedcyon` 配置是旧脚本的做法，没有为它们虚构不存在的独立 YAML。示例显式隔离指标目录，并强制关闭覆盖。

```sh
python train.py --method fedcyon --config config/experiments/CIFAR10_Fedcyon_label01.yaml \
  --gpu 0 --save_name fedcyon_cifar10_b01 --metrics_dir outputs/metrics/fedcyon_cifar10_b01 --overwrite false

python train.py --method fixmatch --config config/experiments/CIFAR10_FedAvg_label01.yaml \
  --gpu 0 --save_name fixmatch_cifar10_b01 --metrics_dir outputs/metrics/fixmatch_cifar10_b01 --overwrite false

python train.py --method fedconsist --config config/experiments/CIFAR10_FedAvg_label01.yaml \
  --gpu 0 --save_name fedconsist_cifar10_b01 --metrics_dir outputs/metrics/fedconsist_cifar10_b01 --overwrite false

python train.py --method cbafed --config config/experiments/CIFAR10_FedAvg_label01.yaml \
  --gpu 0 --save_name cbafed_cifar10_b01 --metrics_dir outputs/metrics/cbafed_cifar10_b01 --overwrite false

python train.py --method rscfed --config config/experiments/CIFAR10_FedAvg_label01.yaml \
  --gpu 0 --save_name rscfed_cifar10_b01 --metrics_dir outputs/metrics/rscfed_cifar10_b01 --overwrite false

python train.py --method isofed --config config/experiments/CIFAR10_Fedcyon_label01.yaml \
  --gpu 0 --save_name isofed_cifar10_b01 --metrics_dir outputs/metrics/isofed_cifar10_b01 --overwrite false
```

第一次先为所选命令加 `--dry-run`；实际小步测试再加 `--num_global_iters 2 --local_iterations 1` 并使用新的 `save_name` 和 `metrics_dir`。旧评估有全训练集梯度诊断，即使只跑两轮也不一定快。

**不要用 `--name_algorithm fedconsist/cbafed/isofed` 机械替换配置中的 `fixmatch`。** 数据包装器按该字段选择弱/强增强返回结构；应保留已核对的配置组合。

### 7.3 其他数据集和 beta

下面是确实存在且 `dataset` 字段相符的代表配置，可替换第 7.2 节对应命令的 `--config`，同时更换输出名称。替换前仍需检查目标方法的数据分支和已知限制。

| 数据集 | FedCyon 代表配置 | 联邦 FixMatch 代表配置 |
| --- | --- | --- |
| CIFAR-10 | `CIFAR10_Fedcyon_label01.yaml` | `CIFAR10_FedAvg_label01.yaml` |
| CIFAR-100 | `CIFAR100_Fedcyon_label01.yaml` | `CIFAR100_FedAvg_label01.yaml` |
| SVHN | `SVHN_Fedcyon_label01.yaml` | `SVHN_FedAvg_label01.yaml` |
| STL-10 | `STL10_Fedcyon_feature01.yaml` | `STL10_FedAvg_feature01.yaml` |
| TinyImageNet | `TINYIMAGENET_Fedcyon_label01.yaml` | `TINYIMAGENET_FedAvg_label01.yaml` |

以上均位于 `config/experiments/`。TinyImageNet 表示已有入口/配置，不应在修复旧评估阻塞前作为可直接完成的实验。

列出所有 YAML 的真实数据集、划分和 beta：

```sh
python - <<'PYCODE'
from pathlib import Path
import yaml
for filename in sorted(Path("config/experiments").glob("*.yaml")):
    config = yaml.safe_load(filename.read_text())
    print(filename.name, "dataset=", config.get("dataset"),
          "partition=", config.get("partition", "<method default>"),
          "beta=", config.get("beta", "<method default>"))
PYCODE
```

### 7.4 旧流程的客户端与划分限制

- 常见 20 客户端配置采用 `[2, 0, 18]`，表示纯有标签 / 混合 / 纯无标签；STL-10 常用 `[1, 0, 19]`。
- 部分方法支持特定的 50/100 客户端分支，但不是所有方法都支持任意客户端数。
- RSCFed 调度保留构造时固定 20 客户端的旧行为；不要通过修改 `num_total_users` 假定它支持任意规模。子组参数为 `S_M`、`S_K`、`S_K_lb`、`selected_mode`，需与客户端协议匹配。
- 旧缓存的名称主要由数据集、客户端数、划分类型、beta 等组成，不包含完整划分种子。命中缓存时直接复用，不会因训练 seed 改变而重新划分。
- FixMatch 额外使用 noise 后缀，新建缓存也不保证与 FedCyon 的客户端样本逐项相同。正式公平对照需要核对实际索引，不能只检查 beta 一致。

### 7.5 CIFAR-10 / SVHN 的 FedPDM、SAGE

先按第 4.5 节准备数据，再在 CUDA 服务器运行：

```sh
python train.py --method fedpdm --config config/experiments/CIFAR10_FedPDM_label01.yaml --gpu 0
python train.py --method sage --config config/experiments/CIFAR10_SAGE_label01.yaml --gpu 0
python train.py --method fedpdm --config config/experiments/SVHN_FedPDM_label01.yaml --gpu 0
python train.py --method sage --config config/experiments/SVHN_SAGE_label01.yaml --gpu 0
```

3 轮检查可追加 `--num_global_iters 3 --num_subset_users 2 --local_iterations 1 --batch_size 2 --eval_interval 1 --save_name <独立名称>`。保留 20 个总客户端以匹配默认显式缓存；不要仅改客户端数而继续读取旧 20 客户端索引。正式配置默认 2000 轮、每轮每客户端 5 步，需按统一对照预算另行确定。

## 8. 多次运行、后台日志与对照实验

### 8.1 ImageNet-100 多 seed

```sh
python train.py --method fedpdm --config config/experiments/ImageNet100_FedPDM_label01.yaml \
  --gpu 0 --runtimes 3 --seed_offset 10 --save_name fedpdm_imagenet100_3runs --overwrite false
```

每次进程中的 run 编号从 1 开始，训练 seed 为 `run + seed_offset`；以上为 11、12、13。默认大多数方法偏移为 10，RSCFed 为 20。ImageNet 的划分仍由 `partition_seed=2026` 控制，三次运行复用相同划分。使用确定性相关设置不等于保证跨设备、跨库版本逐位相同。

### 8.2 旧方法多 seed：独立目录运行

旧评估 `.npy` 多数没有 run 编号，`--runtimes 3` 会覆盖前两次的同名旧指标，虽然新的 `_training.*` 不会混在一起。若需要保留每个 seed 的完整评估结果，推荐分别启动：

```sh
for seed in 11 12 13; do
  python train.py --method fedcyon --config config/experiments/CIFAR10_Fedcyon_label01.yaml \
    --gpu 0 --runtimes 1 --seed_offset "$((seed - 1))" \
    --save_name "fedcyon_cifar10_seed${seed}" \
    --metrics_dir "outputs/metrics/fedcyon_cifar10_seed${seed}" --overwrite false || break
done
```

这是训练 seed 对照，不是重新随机生成客户端划分。不要把已有缓存的 seed 和训练初始化 seed 混为一谈。

### 8.3 保存控制台日志 / 后台运行

当前训练 logger 主要输出到控制台，`get_logger()` 的文件 handler 未启用；不要假定一定生成 `log.txt`。手动重定向 stdout 和 stderr：

```sh
mkdir -p outputs/logs
nohup python -u train.py --method sage --config config/experiments/ImageNet100_SAGE_label01.yaml \
  --gpu 0 --save_name sage_imagenet100_background --overwrite false \
  > outputs/logs/sage_imagenet100_background.log 2>&1 &
```

```sh
tail -f outputs/logs/sage_imagenet100_background.log
```

后台启动本身不表示成功；查看日志以及 `run_1_training.json` 的状态。正式程序不是多 GPU 分布式训练；如使用 `CUDA_VISIBLE_DEVICES=2`，进程内唯一可见 GPU 通常编号为 0，应搭配 `--gpu 0`。没有实现同一实验的 DDP / 多 GPU 并行。

### 8.4 对照前的核对表

比较六种方法时，至少固定：数据及类别列表、实际客户端索引、标注客户端、模型及预训练设置、通信轮数、本地步数、batch 大小、评估频率、训练 seed 集合。显式旧缓存不受新 `partition_seed` 控制；需比较实际索引。RSCFed 还应报告子组重复训练次数。新旧 SVHN 的评估 split 不同，必须统一后才能报告精度对照。

阈值和损失是方法的一部分，应报告各自设置。不同方法 loss 的定义和尺度不同，不能仅根据 train loss 更小判定算法更好；效果应看相同评估协议下的 Accuracy 等指标。资源比较还需统一设备、AMP、workers、评估和诊断设置。

## 9. 结果、耗时、内存与收敛曲线

### 9.1 默认输出布局

以第 6.3 节 FedPDM 实验为例：

```text
outputs/saved_models/fedpdm_imagenet100_runA/
├── resolved_config.json
└── imagenet100_data.json

outputs/metrics/imagenet100/fedpdm_imagenet100_runA/
├── run_1.json
├── run_1.npz
├── run_1_training.json
├── run_1_training.csv
└── run_1_training.npz
```

`resolved_config.json` 保存最终参数；`imagenet100_data.json` 保存数据和标注覆盖摘要。**当前这些文件不是模型 checkpoint**；不要因为目录名为 `saved_models` 就认为里面有可恢复的模型权重。

### 9.2 八种方法共有的训练记录

每个 run 自动保存到 `<metrics_dir>/<dataset>/<save_name>/run_<run>_training.*`，无需额外参数。

| 文件 | 内容 |
| --- | --- |
| `_training.json` | 方法、seed、状态、总耗时、总体内存峰值、逐轮记录及异常信息 |
| `_training.csv` | 每轮一行，适合 pandas / 表格软件 |
| `_training.npz` | 与 CSV 对应的数值数组，可直接用 NumPy 绘图 |

| 关键字段 | 含义 |
| --- | --- |
| `train_loss` | 该轮实际本地 minibatch 的完整优化目标平均值，未做 GradScaler 缩放 |
| `train_steps` | 实际观测的目标次数，不是 AMP 成功更新次数保证 |
| `labeled_train_loss` / `unlabeled_train_loss` / `mixed_train_loss` | 按客户端类型分组的完整目标均值 |
| `training_time_seconds` | 逐轮训练阶段耗时；JSON 顶层为训练阶段累计耗时 |
| `cumulative_training_time_seconds` | 截至本轮的累计训练阶段耗时 |
| `total_time_seconds` | JSON 顶层的总墙钟时间，含服务器初始化、评估及周期性输出 |
| `cpu_peak_rss_mib` / `cpu_tree_peak_rss_mib` | 主进程 / 主进程与子进程 RSS 的采样峰值 |
| `cuda_peak_allocated_mib` / `cuda_peak_reserved_mib` | 选定设备上本进程 PyTorch allocator 的显存峰值 |

CPU 每 0.1 秒和阶段边界采样，可能遗漏短暂尖峰；进程树 RSS 可能重复计入共享页。CUDA 记录不等于整张卡的全部使用量。单位统一为 MiB。CPU 调试时 CUDA 数据缺失，不伪造为 0。

`train_loss` 包含真实使用的损失系数，不是最后一个 batch，也不是全局评估 loss。空轮或非有限目标使用缺失值，并记录相关计数。CBAFed 初始预训练单独记入 JSON 的 `warmup`，计入总训练时间但不混入正式通信轮次曲线。

每完成一轮原子替换文件，包含所有已完成轮次。正常结束为 `completed`，未完成为 `incomplete`，可捕获错误/键盘中断为 `failed` / `interrupted`；强制杀死进程或断电可能只留下 `running` 和此前已经保存的轮次。具体测量边界见 `TRAINING_METRICS.md`。

### 9.3 新索引路径评估文件

适用于 ImageNet-100 六方法，以及 CIFAR-10/SVHN 的 FedPDM、SAGE；均位于 `metrics_dir/<dataset>/<save_name>/`。

- `run_N.json`：数据摘要、每轮训练诊断、客户端/聚合信息、评估历史；新 baseline 额外记录实现变体。
- `run_N.npz`：`eval_rounds`、`testacc`、`top5`、`loss`，其中 `loss` 是验证集 CE，准确率范围是 0–1。
- 第 0 轮是未训练状态，之后按 `eval_interval` 以及最后一轮评估；使用 `eval_rounds` 作横轴，不要按数组下标当通信轮数。
- 采用 `eval()` / 无梯度评估，不执行旧路径的全训练集反向梯度诊断。

### 9.4 旧评估文件：不要和新 train loss 混用

旧方法仍在 `metrics_dir` 根目录保存带算法/划分前缀的 `.npy`：常见包括 `testacc`、`precision`、`recall`、`F1`、`AUC`、`idealloss`，部分数据集另有 `top5`，部分方法保留梯度方差字段。

**旧 `idealloss` 不是本轮优化目标均值，也不能统一称为验证损失**：公共旧评估通常在完整训练集上另算 CE / 梯度诊断；STL-10 还存在占位值。收敛曲线请读取新的 `_training.*`。

旧流程通常在每轮训练前评估，最后一次聚合后不一定再次评估，因此 N 个旧评估点不一定包含“训练完第 N 轮”的性能。未在本次记录改造中静默改变这一行为。

### 9.5 绘制训练收敛曲线

运行第 6.3 节对应实验后，在项目根目录执行：

```sh
python - <<'PYCODE'
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

filename = Path("outputs/metrics/imagenet100/fedpdm_imagenet100_runA/run_1_training.npz")
with np.load(filename, allow_pickle=False) as metrics:
    rounds = metrics["round"]
    losses = metrics["train_loss"]
    valid = np.isfinite(losses)
    plt.plot(rounds[valid], losses[valid], label="FedPDM")
plt.xlabel("Communication round")
plt.ylabel("Mean local training objective")
plt.legend()
plt.tight_layout()
output = filename.with_name("convergence_round.png")
plt.savefig(output, dpi=200)
print(output)
PYCODE
```

时间轴曲线可将横坐标改为 `metrics["cumulative_training_time_seconds"]`。多 seed 的均值/标准差需要读取各 run，按相同轮次对齐，并排除未完成或无效数据；目前不会自动汇总成论文表格。

## 10. 备份与磁盘占用

本项目的大文件主要来自原始图像、共享图像库和历史结果，不是 Python 代码。实验数据已改为共享存储，不能再为每个新实验复制全部原图。

- 旧数据集：同时备份 `data/*.npz` 与其引用的 `data/_shared/<dataset>/`；仅复制客户端索引无法恢复图像。
- ImageNet-100：同时保留原图目录、100 类清单、可选验证映射、共享 manifest 和客户端索引；`source.json` 保留来源信息。
- 旧评估仍要原始数据；共享图像库主要服务训练划分，不等于所有源数据都可删除。
- 新实验结果及配置位于 `outputs/`，被 Git 忽略，迁移机器时要单独复制。
- 不要手动截断、重排 `images.bin`，或删除仍被其他实验引用的共享库。
- `archive/`、`outputs/legacy_results/` 和 `outputs/legacy_logs/` 保存历史资料，不会因为新训练自动清理。

检查旧共享缓存的示例：

```sh
python -m datautils.migrate_partitions verify --partition cifar10_20_noniid-labeldir_0.1.npz
```

迁移命令会在逐份验证成功后删除相应旧缓存，具体行为务必先读 `DATA_STORAGE.md`。ImageNet-100 使用独立的路径索引格式，不要用旧像素缓存迁移工具处理它。

## 11. 常见问题与已知限制

| 现象 | 原因与处理 |
| --- | --- |
| Mac 正式训练提示需要 CUDA | 入口尚不支持 CPU/MPS；本机用 `--dry-run`、`--prepare-data`，正式实验移到 CUDA 服务器 |
| `Model directory already exists` | 使用新 `save_name`；确认确实允许覆盖时才用 `--overwrite true` |
| `--dry-run` 成功，训练仍失败 | dry-run 不导入训练框架、不检查原始数据、缓存可用性或 CUDA |
| 配置叫 CIFAR100/STL10，实际跑 CIFAR10 | 部分旧 `*_Client20.yaml` 的字段有复制残留；检查解析 JSON，不按名字推断 |
| 找不到聚类索引文件 | 旧 CIFAR-10/STL-10 新划分依赖缺失文件；补齐原文件或使用正确已有缓存 |
| FixMatch 找不到其他方法已有的缓存 | 它额外添加 noise 后缀，例如 `..._0.1_0.npy` 查找同名 `.npz`；不能假定与无后缀文件自动对应 |
| `No module named model.nets` | 旧辅助网络构造器引用了缺失包；该分支未全面恢复，不能只凭 `name_net` 字段推断实际网络 |
| TinyImageNet 评估报局部变量错误 | 公共旧评估未创建该数据集的 `eval_ideal_loader`，后续分支却尝试使用；已有配置不能保证完成训练，需要另行修复 |
| SVHN 指标看起来异常 | 当前旧公共“测试”评估实际使用 `split='train'`；未经修正不能作为标准 SVHN test 结果报告。新 FedPDM/SAGE 使用正确 test split，但不能直接和旧 train-split 指标比较 |
| 旧方法很慢或评估影响状态 | 旧评估调用 `model.train()`，并可能全训练集反向计算梯度；未统一为 ImageNet 新路径的纯推理协议 |
| ImageNet 验证目录报缺类 | 检查所选 100 类、目录布局及平铺验证集映射，映射必须用 synset |
| 无标签 loss 为 0 / FedPDM 聚合权重为 0 | 可能是没有伪标签通过阈值；检查 mask 比例、有效样本数和模型预测，不要用真实无标签标签“修复” |
| 开启 pretrained 后结果明显不同 | 引入 ImageNet-1K 监督预训练，已改变监督条件，必须单独报告 |
| 显存 / 内存不足 | 减小 batch / eval batch；必要时重新设计统一客户端规模。ImageNet CPU 客户端状态也占内存，不能只看 GPU |
| 找不到训练日志文件 | 当前 logger 未启用文件 handler；按第 8.3 节重定向 stdout/stderr |
| 重复实验旧 Accuracy 文件只剩最后一次 | 旧 `.npy` 命名无 run 隔离；使用每 seed 独立 `metrics_dir` |
| 想从中断处继续 | 当前没有完整 checkpoint / optimizer 恢复；ImageNet 显式拒绝 `resume/load_path`，旧保留参数也不等于恢复已实现 |

GitHub 版本以当前代码快照建立新的 Git 历史，不包含早期提交历史。本地原有、已压缩且无法直接使用的 `.git` 元数据原样备份在 `archive/git-metadata-*/`，未上传。仓库包含源码、配置、依赖文件和文档；`data/`、`outputs/`、`archive/`、虚拟环境、模型权重和本地凭据文件均不纳入版本控制。克隆仓库后需单独准备数据；文档中的本地测试日志和历史归档路径不会随仓库下载。这次发布不修改上述旧算法问题。

## 12. 验证范围与文档导航

### 已做的检查

- 当前 Python 源码语法、107 份 YAML 的可解析性，以及指定 18 个组合的配置和服务端导入。
- 2026-10-20 补齐 7 个组合后，18 个组合重新完成 3 轮 CPU 核心链路冒烟测试；新增目标函数、聚合、缓存复用另有数值检查。边界和证据见 `ADAPTER_TEST_REPORT.md`，不代表正式 CUDA/AMP 或官方测试集全量验证通过。
- ImageNet-100 路径索引、IID/Dirichlet 划分、缓存复用和异常索引拒绝。
- FedCyon、FedPDM、SAGE-Pure 在合成 100 类图片上的两轮 CPU 联调。
- FedPDM/SAGE-Pure 用真实图片和 ResNet18，各做两个客户端的两轮 CPU 小步训练；调试阈值设为 0 以覆盖无标签反向分支。
- 时间/内存/loss 记录的均值和计数、空轮与非有限值、异常/中断保存、JSON/CSV/NPZ 一致性。
- 六条旧服务端真实循环配合桩客户端的记录检查，覆盖 CBAFed 预训练、IsoFed 双阶段和 RSCFed 多子组。
- CUDA 同步与峰值 API、重复运行清理做过模拟检查，不等于真实 GPU 执行。

### 尚未完成

真实 CUDA/AMP 全流程验证、六种旧方法与全部旧数据集的端到端回归、全量训练收敛、多 seed 统计和论文精度复现。当前没有可直接运行的统一 `pytest` 测试套件；此前的小步检查不应被描述为全量实验。

### 进一步阅读

| 文档 | 内容 |
| --- | --- |
| `BASELINES.md` | FedPDM / SAGE-Pure 目标与聚合、方法来源、工程默认值、与原论文的差异 |
| `ADAPTER_TEST_REPORT.md` | 7 个新增适配、18 组合 3 轮 CPU 测试、算法差异及 GPU 复跑命令 |
| `TRAINING_METRICS.md` | 耗时和内存的统计边界、实际 train loss、缺失值、中断保存及绘图 |
| `DATA_STORAGE.md` | 共享图像设计、迁移、校验、导出与备份 |
| `archive/README.md` | 历史文件的用途与整理范围 |

推荐工作顺序：**安装环境 → dry-run → 准备数据/检查划分 → 两轮 GPU 小步检查 → 完整实验 → 多 seed 对照 → 汇总精度、耗时、内存和收敛曲线。**
