# 多分类量子卷积神经网络的经典—量子接口设计

[**English**](README.md) | **简体中文**

本仓库为论文 **《Classical–Quantum Interface Design for Multi-Class Quantum Convolutional Neural Networks》** 的配套实验代码与配置文件。

代码实现了一个递进式实验框架，用于研究**仅依赖极少经典辅助**条件下、多分类**量子卷积神经网络（QCNN）**的**经典—量子接口**。整体流程为：数据降维与归一化 → 角度编码电路 → 参数化纠缠 Ansatz → 单比特 Pauli-*Z* 测量 → 经典全连接读出。所有量子电路均在 **MindQuantum**（`mqvector` 态矢量模拟器）上模拟，并借助 **MindSpore** 训练。

---

## 目录结构

```
.
├── Experiment I Influence of different Ansatz structures and layer numbers/
│   ├── Influence of different Ansatz structures and layer numbers.ipynb
│   └── results_per_exp/                     # 9 × (_history.pkl + _model.ckpt)，L = 1, 2, 3
├── Experiment II Influence of data dimensionality reduction methods/
│   ├── UMAP.ipynb
│   ├── autoencoder.ipynb
│   └── *.npz                                # UMAP / 自编码器结果（PCA 复用实验一的结果）
├── Experiment III Effect of the pre-encoding scaling interface/
│   ├── PCA.ipynb   UMAP.ipynb   autoencoder.ipynb
│   └── *_scaling_results*.npz
├── Experiment IV Comparison of post-measurement weighting layers/
│   ├── PCA.ipynb   UMAP.ipynb   autoencoder.ipynb
│   └── *_weight_relu_results*.npz
├── Ablation study necessity of the quantum circuit/
│   ├── {PCA, UMAP, autoencoder} × {冻结 Ansatz, 仅编码器, MLP, SVM}  → 12 个 notebook
│   └── 12 × *.npz
└── Dimensionality reduction on CIFAR-10/
    ├── {PCA, UMAP, autoencoder}_cifar-10_{2,10}.ipynb     → 6 个 notebook
    └── cifar10_{binary,10class}_*_results.npz             → 6 个结果文件
```

**总计**：27 个 notebook、26 个 `.npz` 结果文件、9 份训练历史（`.pkl`）、9 个模型检查点（`.ckpt`）。

---

## Notebook 与论文的对应关系

| 文件夹 | 论文节次 | 主要产出 |
|---|---|---|
| Experiment I | §4.1 实验一 | 图 4；门数与深度分析 |
| Experiment II | §4.2 实验二 | 表 3 |
| Experiment III | §4.3 实验三 | 表 4、图 5 |
| Experiment IV | §4.4 实验四 | 表 5、图 6 |
| Ablation study | §4.5 消融研究 | 表 6、图 7 |
| Dimensionality reduction on CIFAR-10 | §4.6 CIFAR-10 上的降维对比 | 表 7、图 8 |

---

## 运行环境

- Python 3.9+
- **MindSpore** 与 **MindQuantum**（CPU 版本即可）
- `numpy`、`scipy`、`scikit-learn`、`matplotlib`
- `umap-learn`（UMAP 相关实验）
- `torchvision`（仅用于下载 MNIST / CIFAR-10；消融实验的 notebook 通过 `urllib` 直接下载 MNIST）

```bash
pip install mindspore mindquantum numpy scikit-learn matplotlib umap-learn torchvision
```

全部实验均在 **CPU** 上完成（`MS_DEVICE = "CPU"`），无需 GPU；全程使用 MindQuantum 的 `mqvector` 模拟器。

---

## 数据集

- **MNIST**：由 notebook 自动下载，按 **8:2** 划分为训练集 / 测试集（56 000 / 14 000）。
- **CIFAR-10**：通过 `torchvision.datasets.CIFAR10` 自动下载，评估两个任务：
  - *二分类*：飞机与汽车两类，每类 640 张训练 / 100 张测试 → 1 280 / 200；
  - *十分类*：全部十类，每类 640 / 100 → 6 400 / 1 000。

两个数据集均先展平（MNIST 为 `784` 维、CIFAR-10 为 `3072` 维）并缩放到 `[0, 1]`，降至 **8 维**后再缩放到 `[-1, 1]`，随后进行角度编码。所有变换参数**仅在训练集上拟合**。

---

## 使用方法

打开所需的 notebook 并运行全部单元格即可。每个 notebook 均自成闭环：加载数据 → 构建电路 → 训练模型 → 输出指标 → 将结果写入同目录下的 `.npz` 文件。

```bash
jupyter notebook
```

运行时间主要取决于量子模拟。CIFAR-10 与消融实验的 notebook 最轻量，而实验三 / 实验四（三种预处理 × 三种接口变体）耗时最长。

---

## 关键配置

| 参数 | 取值 |
|---|---|
| 量子比特数 | 8 |
| 编码方式 | Hadamard 层 → `RY` 角度编码 → 砖墙式 `CNOT` |
| Ansatz（默认） | ChainEntangled（`sequential_su4`），**L = 2** |
| 测量 | 对全部 8 个量子比特测量单比特 Pauli-*Z* 期望值 |
| 读出 | 全连接 8 → 10（`Normal(0.02)` 初始化，偏置为零） |
| 优化器 | Adam，学习率 `0.01` |
| 批大小 | 32 |
| 损失函数 | 作用于 logits 的稀疏类别交叉熵 |
| 训练轮数 | 150（不启用早停） |
| 随机种子 | 42 |
| 并行工作器 | 2 |
| 设备 / 模拟器 | CPU / MindQuantum `mqvector` |

**降维方法配置**

| 方法 | 配置 |
|---|---|
| PCA | 8 个主成分，直接作用于已缩放到 `[0, 1]` 的特征 |
| 自编码器 | `输入 → 256 → 64 → 8` 瓶颈层；隐藏层采用 HeNormal 初始化 + ReLU；MSE 损失、Adam `lr = 0.001`、批大小 128、训练 50 轮；训练后参数**冻结**，仅作为固定特征提取器 |
| UMAP | `n_components = 8`、`n_neighbors = 10`、`min_dist = 0.5`、`random_state = 42` |

---

## 命名约定

代码中三种 Ansatz 的名称与论文中的称呼并不一致：

| 代码中的名称 | 论文中的名称 |
|---|---|
| `sequential_su4` | **ChainEntangled** |
| `center_sequential_su4` | **CenterEntangled** |
| `so4` | **LightEntangled**（仅使用 RY 与 RYY） |

结果文件名带 `nn10_md0.5` 后缀者，对应 UMAP 取 `n_neighbors = 10`、`min_dist = 0.5`。

实验二中的 **PCA** 基线未重新运行：实验一已将预处理方法固定为 PCA，因此直接复用其结果。

---

## 输出文件

每个 notebook 会写出一个 `.npz` 归档（实验一另为每组配置写出 `_history.pkl` 与 `_model.ckpt`），其中包含逐轮训练损失、逐轮测试准确率以及整个训练过程中的最佳测试准确率。具体键名可参见各 notebook 的最后一个单元格。

---

## 主要结果

| 实验 | 结果 |
|---|---|
| 实验一 | ChainEntangled L = 2：**84.52%**（CenterEntangled 76.73%、LightEntangled 63.12%） |
| 实验二 | PCA **84.52%** / 自编码器 **91.17%** / UMAP **94.26%** |
| 实验三 | 缩放接口未带来一致增益；PCA 下编码前侧的全连接缩放降至 76.36% |
| 实验四 | 同一接口移至测量端后，PCA 下回升至 84.54%，自编码器下为 91.85% |
| 消融 | 移除或冻结 Ansatz 均导致准确率下降；同样 8 维特征上，一个规模不大的经典 MLP 可达 88.94–94.78% |
| CIFAR-10 | 二分类：PCA 83.50% / 自编码器 82.00% / UMAP 81.50%；十分类：29.90% / 28.80% / 31.00% |

以上数字仅为便于查阅而列出；最终数值、协议与讨论请以论文为准。

---

## 可复现性说明

- 全程固定随机种子为 `42`。
- 每种实验配置**仅运行一次**，因此所报告的准确率不含运行间方差估计。零点几个百分点的差异（例如 CIFAR-10 上三种降维方法之间）应视为指示性而非可靠差异。

---

## 引用

```bibtex
@article{liu_qcnn_interface,
  title   = {Classical--Quantum Interface Design for Multi-Class Quantum Convolutional Neural Networks},
  author  = {Liu, Zihao and Zheng, Chengyong and Wang, Jinge},
  journal = {},
  year    = {}
}
```

*（论文正式发表后请补全期刊名与年份。）*

---

## 许可协议

*待补充（例如 MIT / Apache-2.0）。*

## 联系方式

- 刘子豪（Zihao Liu）—— 五邑大学数学与计算科学学院，中国江门
- 欢迎通过 GitHub Issue 提问或反馈。
