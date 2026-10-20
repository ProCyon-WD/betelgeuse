# 三数据集新基线：FedPDM / SAGE-Pure

这两个入口支持 `imagenet100`、`cifar10`、`svhn`。它们共用客户端索引、抽样、
训练调度和流式评估，不复用 FedCyon 的类别重加权损失或 TA 聚合。
ImageNet-100 使用 torchvision ResNet18/50；CIFAR-10 使用项目的 32 像素 ResNet18，
SVHN 使用 VGG19-BN。小图复用相同 FedPDM/SAGE objective 与 aggregation，不改写旧六方法。

## 运行

在项目根目录、已安装 `requirements-cuda.txt` 的 CUDA 环境中运行：

```sh
python train.py --method fedpdm --config config/experiments/ImageNet100_FedPDM_label01.yaml
python train.py --method sage --config config/experiments/ImageNet100_SAGE_label01.yaml
python train.py --method fedpdm --config config/experiments/CIFAR10_FedPDM_label01.yaml
python train.py --method sage --config config/experiments/CIFAR10_SAGE_label01.yaml
python train.py --method fedpdm --config config/experiments/SVHN_FedPDM_label01.yaml
python train.py --method sage --config config/experiments/SVHN_SAGE_label01.yaml
```

Mac 可在命令末尾加 `--dry-run` 检查参数，或加 `--prepare-data` 准备索引；
正式 CLI 训练仍要求 CUDA。不要省略 `--method`，方法与 YAML 的 `name_algorithm`
不一致会报错，避免把 FedPDM/SAGE 配置误跑成 FedCyon。

默认复用 `data/datasource/imagenet100`；服务器可加 `--datadir /datasets/imagenet100`。
两份配置保持原 FedCyon 的类别表、划分 seed、beta、20 个客户端 / 2 个有标签客户端、
ResNet18、无预训练、200 轮、每轮每客户端 5 个本地优化步骤以及 batch size。
这是一组可运行的工程起点，不是论文最优超参数或复现精度承诺。

小图四份配置默认显式复用 `data/<dataset>_20_noniid-labeldir_0.1.npz`，
这时配置中的划分 seed 不重新作用于已有索引。用 `--partition_file none` 可改成从原始
训练集按 seed 生成索引，仍去重到共享图像库，不依赖旧聚类文件。已有训练缓存不含独立
测试集：需准备 `data/datasource/<dataset>/` 或设置 `--datadir`；允许下载时加
`--download_data true`。`--prepare-data` 在创建客户端前检查划分和原始测试数据。
新 SVHN 评估使用 test split；旧方法的 train-split 评估问题未在本次改写，不能直接比较两者精度。

输出位于 `outputs/metrics/<dataset>/<save_name>/run_<run>.json` 和同名 `.npz`。
JSON 的 `implementation` 明确标识实现变体；`training[].clients` 保存每个客户端
损失、伪标签通过率、聚合计数和归一化聚合权重。准确率范围是 0–1。

每个 run 另存 `_training.json` / `_training.csv` / `_training.npz`，记录总/逐轮
训练耗时、CPU/CUDA 内存峰值及真实优化目标的逐轮 `train_loss`。画训练收敛曲线使用
`_training.npz`，不要把原评估 NPZ 的 `loss` 当作训练 loss。完整口径见 `TRAINING_METRICS.md`。

## FedPDM

来源：Dongping Liao 等，*Progressive Distribution Matching for Federated
Semi-Supervised Learning*，AAAI 2025，Algorithm 1 和公式 (4)–(7)。
论文：https://ojs.aaai.org/index.php/AAAI/article/download/32551/34706

独立实现，没有导入第三方训练代码。结果标记为 `fedpdm_pure_reimplementation`。

- 每个参与客户端从下发模型初始化在线模型和 EMA teacher；本地优化后更新 teacher，
  上传 **EMA 模型**。EMA 参数按 `ema_m` 更新，BatchNorm 等 buffers 从在线模型复制。
- 无标签客户端仅依据 teacher 对弱增强的预测维护自己的类别先验 EMA，绝不读取其
  ground-truth labels。有标签客户端仅用自己的标签统计类别先验。
- 每个无标签客户端维护 `N_client × num_classes` 的 CPU float32 预测缓存，按本地
  样本索引更新，跨轮保留；初始预测和先验均为均匀分布。
- 在 log 域对 `P ** pdm_sinkhorn_power` 做 Sinkhorn 行列缩放，行目标为 1、列目标为
  `N_client * prior`，避免行列总质量不一致；最后将每行归一化。有限次迭代的列边际是近似值。
  对校准后的预测按 `p_cutoff` 筛选并取 argmax，监督强增强分支。
- 有标签与无标签损失都在 logits 上加入 `strength * log(prior)`；零频率以 `1e-6`
  平滑。`strength` 在 `pdm_rampup_rounds` 个全局轮次中线性增加到 `pdm_logit_strength`。
- 聚合原始权重：有标签端为本地样本数；无标签端为 **本轮实际访问且最近一次通过阈值的
  不重复样本数**。重复采样不重复计数，不用 `通过率 × 数据集大小` 外推。
  通过率为零时权重为零；整体无贡献时保持原全局模型，避免除零。

论文以完整 local epochs 训练；本项目为保持现有对照预算采用固定 `local_iterations`，
因此默认 5 步时可能只访问无标签客户端的一小部分样本，聚合会偏重有标签端。
这点及置信阈值 0.95 导致的早期零通过率应通过日志观察，不要通过偷看真实标签修正。

以下是显式的工程默认值，未宣称全部来自论文附录：
`ema_m=0.99`、`pdm_prior_ema=0.9`、`pdm_sinkhorn_power=3`、
`pdm_sinkhorn_iters=30`、`pdm_logit_strength=1`、`pdm_rampup_rounds=20`。
采用与现有配置一致的固定学习率；未照搬原论文 Pure 场景的分组学习率和 cosine 调度。

## SAGE-Pure

来源：*Mind the Gap: Confidence Discrepancy Can Guide Federated Semi-Supervised
Learning*，CVPR 2025。对照作者仓库：
https://github.com/Jay-Codeman/SAGE/tree/e90fa96e0766fc2d6e4ea64625b7eec8aa1fd506

根据作者训练逻辑独立编写；没有复制其模型、数据划分或训练脚本。
**原实现是每客户端混合标注，本项目是纯有标签 / 纯无标签客户端；因此明确标为
`sage_pure_adapted`，论文表格应写 SAGE-Pure (adapted)，不能称为原设定的原版复现。**

- 有标签端使用弱增强图像的普通 CE，不使用 FedCyon 的类别损失。
- 无标签端同时获取在线模型和本轮冻结的全局 teacher 的弱增强预测。
  取置信度差 `gap = abs(conf_local - conf_global) + 1e-6`（上限 1），
  用 `exp(-log(2) / sage_half_gap * gap)` 混合两者的 one-hot 伪标签。
- 当局部预测不过阈值时使用全局伪标签；局部或全局任意一个过阈值即可保留样本。
  强增强输出对修正后的软标签计算 KL loss；损失对整个 batch 取均值。
  默认 `sage_half_gap=0.05` 与作者代码固定的敏感度一致。
- teacher 不反向传播、不更新 BatchNorm、不参与 EMA；每轮从广播模型重新建立。
  服务端上传在线模型并按本地样本数做 FedAvg，不使用 TA 或有效伪标签数加权。

## 共同工程约定与比较边界

- 只有当前客户端及其 teacher 在计算设备上。客户端模型和独立 optimizer 状态保留在 CPU，
  teacher 在客户端结束后释放；不会像参考 SAGE 脚本一样让多个客户端共用一个 optimizer。
- 图像始终只存一份。FedPDM 的预测缓存不是图像；100 类、126,689 个样本的完整 float32
  预测矩阵上限约 50.7 MB，实际只为已参与的无标签客户端分配。Sinkhorn 另有工作内存，
  每个本地步骤成本约 `O(N_client * num_classes * iterations)`，在 CPU 上执行。
- 浮点参数和 buffers 按权重平均；整数 buffers 取最大权重客户端的值，平权时取先到者。
- 两种新方法的有标签端使用弱增强；原 FedCyon 有标签端仍使用基础视图。
  若研究目标是严格隔离 objective/aggregation 的贡献，需要另做统一增强的消融实验。
- 目前不新增 Mixed 标签协议，不支持断点恢复，也没有修改旧六方法的数据切分。
- 当前已下载数据的默认划分中 `n02701002` 未出现在有标签客户端；所有方法共用此划分，
  不在 baseline 内偷偷补标签。若要保证100类都有监督，应统一更改划分协议并重跑所有方法。
- 先检查 `aggregation_weight`、伪标签通过率和损失是否正常，再进行多随机种子完整实验；
  不用最终评估集调超参数。CPU smoke test 不等于完成 CUDA/AMP 或论文精度复现。

## 已完成的检查

- Sinkhorn 的行和/列边际、极端概率与缺失类别数值稳定性。
- SAGE 与作者伪标签/KL 公式的数值对照、teacher 无梯度、全拒绝掩码。
- 加权聚合、零贡献客户端、整数 buffers 和非法权重拒绝。
- 两种新方法和原 FedCyon：合成100类数据上各两轮 CPU 端到端，包含结果保存。
- 两种新方法：真实 ImageNet-100 图片、真实 ResNet18，各两个客户端、两轮、
  每客户端每轮一个优化步骤；无标签 loader 的 targets 设为 None，验证无需真实标签。
  此项临时使用阈值0以覆盖无标签反向路径；另行检查阈值1时全拒绝且聚合不除零。
- ImageNet-100 六方法可复用相同真实数据清单和划分；活动 Python 源码通过语法解析，107 份 YAML 可解析。
- 两种新方法在 CIFAR-10、SVHN、ImageNet-100 上各通过 3 轮 CPU 小步训练；小图评估使用缓存诊断样本，不是官方测试集精度。详见 `ADAPTER_TEST_REPORT.md`。
- 这些检查尚不包含真实 CUDA/AMP、全量训练收敛、多随机种子统计或论文表格精度复现。
