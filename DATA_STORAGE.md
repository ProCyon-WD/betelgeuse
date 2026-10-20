# 共享图像与实验划分

训练缓存现在按图像内容去重。同一数据集的相同像素只保存一次；每个实验保留客户端编号、样本索引、标签及校验值。客户端顺序、样本顺序、重复样本、标签值、数组形状和 dtype 均保留，不重新随机划分数据。

## 文件布局

```text
data/
  _shared/
    cifar10/
      images.bin
      metadata.json
      index.sqlite3
      .lock
    stl10/
      ...
  cifar10_20_noniid-labeldir_1.npz
  stl10_20_noniid-featuredir_0.4.npz
  migration_report.json
```

- `images.bin`：按固定形状和 dtype 连续保存的图像，通过只读 memory map 加载。
- `metadata.json`：共享图像的形状、dtype、格式版本。
- `index.sqlite3`：图像内容 SHA-256 到图像编号的索引，新实验追加数据时用于去重。
- 每个 `.npz`：压缩保存各客户端的图像编号、原始标签和校验值；不包含图像像素，不依赖 pickle。
- `migration_report.json`：记录迁移源文件、迁移前后大小、客户端/样本数量及验证结果。

图像按字节内容而不是标签去重，因此同一图像在不同实验中的不同标签仍会保留。经过噪声或其他预处理后像素不同的图像不会被误合并。

## 训练端

FedCyon/Balanced、FixMatch、FedConsist、CBAFed、RSCFed 和 IsoFed 的数据读写都已接入 `datautils.partition_store`。

```python
from datautils.partition_store import load_partition, save_partition

partition = load_partition("data/cifar10_20_noniid-labeldir_1.npy")
client_images = partition[0]["data"]
first_image = client_images[0]
```

调用方可以继续传入旧的 `.npy` 路径；文件不存在时会查找同名 `.npz` 或旧 `.npy.gz`。新实验调用 `save_partition(filename, data_map)` 会写入同名 `.npz`，并复用共享图像。默认缓存目录为项目根目录的 `data`，可用 `BETELGEUSE_DATA_DIR` 指向其他位置。

图像对象支持长度、整数/切片索引、`shape`、`dtype`、`ndim`、深复制及进程间 pickle。复制客户端对象不会复制整个图像库；`np.asarray(client_images)` 才会显式物化该客户端图像。

不要再使用 `np.load(old_path, allow_pickle=True).item()` 直接读取迁移后的实验缓存。旧的独立分析脚本应改用 `load_partition`。

## 迁移和验证

仅需 NumPy 和 Python 标准库，不需要 PyTorch、TensorFlow 或 GPU。写入锁使用 `fcntl`，适用于本项目原有的 Linux 环境及 macOS。

在项目根目录运行：

```sh
python -m datautils.migrate_partitions migrate
python -m datautils.migrate_partitions verify
```

也可以指定外部数据目录或单个实验：

```sh
python -m datautils.migrate_partitions migrate --data-dir /path/to/data
python -m datautils.migrate_partitions verify --partition cifar10_20_noniid-labeldir_1.npz
```

迁移流程：

1. 加载一份旧 `.npy` 或 `.npy.gz`，检查客户端数据结构。
2. 按图像 SHA-256 复用或追加共享图像，并同步图像文件和 SQLite 索引。
3. 写入临时 `.npz`，重新读取它，逐客户端逐块比较每个图像和标签的字节、顺序、形状与 dtype。
4. 校验成功后原子发布索引文件，再重新读取验证，最后删除这一份旧缓存。
5. 写入迁移报告。已经迁移的文件会重新验证，不会重新保存像素。

中断后可重跑迁移命令。未通过验证的旧缓存不会被删除；写入失败留下的未提交图像尾部会在下次写入时回收。程序不保留完整旧文件副本，以实际释放磁盘空间。一次仅加载一个旧实验，但迁移最大文件仍需要数 GiB 可用内存及容纳一份共享图像库的额外磁盘空间。

旧缓存是含 pickle 的 NumPy 对象文件，只能迁移可信来源的数据。新 `.npz` 使用 `allow_pickle=False` 读取。

## 恢复为旧格式

需要交给尚未适配的旧程序时，可以把一个实验导出为原来的字典 `.npy`：

```sh
python -m datautils.migrate_partitions export \
  --partition cifar10_20_noniid-labeldir_1.npz \
  --output /path/to/export/cifar10_20_noniid-labeldir_1.npy
```

导出会重新占用图像空间，且拒绝覆盖已有输出文件。

## 备份和删除注意事项

- 搬迁或备份时必须同时保留实验 `.npz` 和对应的 `_shared/<dataset>` 目录；引用使用相对路径，整个数据目录可以搬迁。
- 单个实验索引可以独立删除，但只要还有实验依赖某个图像库，就不能删除该图像库。
- 不要修改、重排或单独截断 `images.bin`。共享库采用追加写入，不做自动垃圾回收。
- `verify` 会检查所有实验引用的图像及标签校验值；它不依赖原缓存仍然存在。
- 这次改动只处理训练划分缓存。原始数据集下载位置、评估数据加载、结果输出的历史绝对路径及已有压缩源码问题没有一并改动。

## ImageNet-100 路径索引后端

ImageNet-100 不写入 `images.bin`，也不预先将原图转为固定尺寸数组；原始压缩图像只保留在 `--datadir` 指定的目录中，按需读取并转为 RGB。

```text
data/
  _shared/imagenet100/<manifest_sha256>.json
  partitions/imagenet100/<partition_metadata_sha256>.npz
```

- 共享 JSON 保存有序100类表、训练/验证的相对路径和类别。清单 SHA-256 保护类表、路径、标签及顺序，**不是每张图像内容的校验值**；原地替换同名图像不会改变该摘要。
- 每次准备时只扫描目录/映射，不解码图像。类别子集或文件列表变化会产生新的清单，不会继续使用旧索引。
- 每个实验 `.npz` 仅保存各客户端的全局训练样本索引，以及清单摘要、客户端数、标注客户端数、seed、beta、划分类型等元信息，不重复保存图像和标签。通过 `allow_pickle=False` 读取并检查索引无重复、无遗漏。
- `partition_seed` 固定数据划分，不随重复训练的随机种子变化；类表顺序决定标签映射。`BETELGEUSE_DATA_DIR` 同样可以改变缓存根目录。
- FedCyon、FedPDM、SAGE-Pure 在相同数据参数下复用同一份清单和客户端索引。FedPDM 的 CPU 预测缓存只是训练内存中的类别概率，不保存或复制图像，也不产生另一份数据划分。
- 原图目录可以整体搬迁，通过新的 `--datadir` 使用，清单中的相对路径和客户端索引不变。
- 这一后端使用独立格式及划分子目录，不会被旧图像库工具的根目录批量扫描混入；不使用其迁移、导出或逐像素 `verify` 命令。备份需要同时保留原始图像、类别/验证映射文件、共享清单和实验索引。

运行方法及验证集平铺目录的映射格式见 `README.md` 的 ImageNet-100 章节。
