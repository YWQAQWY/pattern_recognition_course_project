# 模式识别课程设计 — Pattern Recognition Course Project

## 项目简介

本项目为模式识别课程设计，实现了两种经典的**降维+分类**方法在不同数据集上的应用：

| 任务 | 降维方法 | 分类器 | 数据集 |
|------|----------|--------|--------|
| Task 1 | LDA (Linear Discriminant Analysis) | MLP (Multi-Layer Perceptron) | MNIST 手写数字 (70,000 样本, 784 维, 10 类) |
| Task 2 | PCA (Principal Component Analysis) | SVM (Support Vector Machine, RBF Kernel) | Wine 葡萄酒 (178 样本, 13 维, 3 类) |

## 实验流程 (Pipeline)

```
原始数据
  │
  ├──→ Hold-out 测试集 (20%, stratify)
  │
  └──→ 训练+验证集 (80%)
        │
        └──→ K 折交叉验证 (K=5, StratifiedKFold)
              │
              ├── Fold i: Train ──→ Val
              │    1. 归一化 (StandardScaler fit on Train, transform Val) ← 防泄漏
              │    2. 降维 (LDA/PCA fit on Train, transform Val)
              │    3. 训练分类器 (MLP/SVM)
              │    4. 验证评估 (Accuracy, Precision, Recall, F1)
              │
              └──→ CV 汇总: mean ± std
  │
  └──→ 全量 Train+Val 重新训练
        │
        └──→ Hold-out 测试集评估
              ├── 混淆矩阵 (Confusion Matrix)
              ├── 决策面可视化 (Decision Boundary)
              └── 综合性能指标 (Accuracy, Precision, Recall, F1)
```

### 关键设计要点

- **防止数据泄漏**: 归一化参数 (`StandardScaler`) 和降维参数 (LDA/PCA) 均只在训练集上 `fit`，验证集与测试集仅执行 `transform`
- **K 折交叉验证**: 采用 `StratifiedKFold` (K=5) 保持每折的类别分布与原始数据一致，结果以均值 ± 标准差呈现
- **Hold-out 测试集**: 20% 数据自始至终不参与训练或验证，仅在最终模型上用于独立测试

## 决策面可视化方法

### 核心挑战

分类器在 **高维空间** 训练（LDA: 9 维, PCA: 8 维），但可视化只能在 **2D 平面** 上呈现。不存在将高维决策面完整"投影"到 2D 的方法——任何 2D 展示都只是高维空间的一个切片。

### 可行方法对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **均值切片法** (本项目采用) | 前 2 个分量做网格，其余维度固定为训练集均值 | 直接使用最终训练好的模型，实现简单 | 只展示高维空间的一个 2D 切片 |
| 直接 2D 重训练 | 降维到 2 维后重新训练分类器 | 决策面完全精确 | 模型改变，性能下降，与最终模型不同 |
| 多切片图 | 固定第 3 分量取多个值 (-1σ, 0, +1σ)，画多个子图 | 展示决策面随第 3 维的变化趋势 | 图多，仍不能覆盖所有维度 |
| 置信度热力图 | 用 `predict_proba` / `decision_function` 计算每个网格点的分类概率/距离 | 展示"软"决策面，信息更丰富 | 仍是 2D 切片 |

### 本项目实现：均值切片法

**原理**：对于降维后的数据空间，选取信息量最大的前 2 个分量（LD1/LD2 或 PC1/PC2）建立 2D 网格；其余维度取训练集上的均值，将每个网格点补齐为完整的特征向量后送入分类器预测。

**Task 1 (LDA + MLP, MNIST) 具体做法**：

```
1. X_trainval → LDA(9-dim) → 取前 2 列 → 确定网格范围 [x_min, x_max] × [y_min, y_max]
2. 取后 7 列的均值向量 mean_remaining (近似为 0)
3. 对每个网格点 p = (a, b):
     p_full = [a, b, mean_remaining[0], ..., mean_remaining[6]]  # 9 维
     label = mlp_final.predict(p_full)
4. 用不同颜色填充网格，叠加测试集散点
```

**Task 2 (PCA + SVM, Wine) 具体做法**：

```
1. X_trainval → PCA(8-dim) → 前 2 列 → 网格范围
2. 后 6 列均值在 PCA 中心化后为 0，直接补零
     p_full = [a, b, 0, 0, 0, 0, 0, 0]  # 8 维
     label = svm_final.predict(p_full)
3. contourf 填充决策区域，scatter 叠加测试样本
```

**合理性说明**：
- LDA 和 PCA 变换后，数据已中心化（整体均值 = 0），其余维度的均值 ≈ 0，因此用均值填充是信息损失最小的做法
- 前 2 个分量解释了最大方差/判别信息，在此平面上的决策面切片最具代表性
- 该方法是降维+分类 pipeline 决策面可视化的学术论文中最常见的做法，例如 [t-SNE 论文](https://lvdmaaten.github.io/tsne/) 等高维可视化工作均采用类似策略

## 项目结构

```
pattern_recognition/
├── src/
│   ├── ldaj2_mlp.py      # Task 1: LDA + MLP on MNIST
│   └── pca_svm.py        # Task 2: PCA + SVM on Wine
├── results/
│   └── figures/           # 可视化输出
│       ├── lda_mlp_confusion_matrix.png      # LDA+MLP 混淆矩阵
│       ├── lda_mlp_decision_boundary.png     # LDA+MLP 决策面
│       ├── lda_mlp_metrics_summary.png       # LDA+MLP 性能指标汇总
│       ├── pca_svm_confusion_matrix.png      # PCA+SVM 混淆矩阵
│       ├── pca_svm_decision_boundary.png     # PCA+SVM 决策面
│       ├── pca_svm_pca_projection.png        # PCA+SVM 投影散点图
│       └── pca_svm_metrics_summary.png       # PCA+SVM 性能指标汇总
├── test.py                # 环境验证脚本
└── README.md
```

## 实验环境

| 依赖 | 版本 |
|------|------|
| Python | 3.12.3 |
| scikit-learn | 1.8.0 |
| NumPy | 2.4.4 |
| Matplotlib | 3.10.9 |
| Seaborn | 0.13.2 |

### 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install scikit-learn numpy matplotlib seaborn
```

### 运行实验

```bash
# Task 1: LDA + MLP (MNIST)
python src/ldaj2_mlp.py

# Task 2: PCA + SVM (Wine)
python src/pca_svm.py
```

## 实验结果概要

### Task 1: LDA + MLP on MNIST

| 指标 | 交叉验证 (5-Fold) | 测试集 (Hold-out) |
|------|-------------------|-------------------|
| Accuracy | 0.9148 ± 0.0018 | 0.9215 |
| Precision (macro) | 0.9141 ± 0.0016 | 0.9204 |
| Recall (macro) | 0.9138 ± 0.0017 | 0.9204 |
| F1-score (macro) | 0.9137 ± 0.0018 | 0.9203 |

- LDA 降至 9 维 (最大判别维度 = n_classes - 1)
- MLP 结构: 输入 9 → 128 → 64 → 10, 激活函数: ReLU, 学习率: 0.001, 迭代: 100

### Task 2: PCA + SVM on Wine

| 指标 | 交叉验证 (5-Fold) | 测试集 (Hold-out) |
|------|-------------------|-------------------|
| Accuracy | 0.9862 ± 0.0169 | 0.9722 |
| Precision (macro) | 0.9865 ± 0.0172 | 0.9778 |
| Recall (macro) | 0.9878 ± 0.0151 | 0.9667 |
| F1-score (macro) | 0.9865 ± 0.0167 | 0.9710 |

- PCA 降至 8 维 (累积解释方差比: 92.09%)
- SVM 核函数: RBF
