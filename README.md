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
| Accuracy | 0.9129 ± 0.0020 | 0.9166 |
| Precision (macro) | 0.9119 ± 0.0020 | 0.9154 |
| Recall (macro) | 0.9119 ± 0.0021 | 0.9156 |
| F1-score (macro) | 0.9117 ± 0.0021 | 0.9154 |

- LDA 降至 9 维 (最大判别维度 = n_classes - 1)
- MLP 结构: 输入 9 → 128 → 64 → 10, 激活函数: logistic, 学习率: 0.001, 迭代: 50

### Task 2: PCA + SVM on Wine

| 指标 | 交叉验证 (5-Fold) | 测试集 (Hold-out) |
|------|-------------------|-------------------|
| Accuracy | 0.9717 ± 0.0267 | 0.9444 |
| Precision (macro) | 0.9729 ± 0.0238 | 0.9484 |
| Recall (macro) | 0.9763 ± 0.0226 | 0.9484 |
| F1-score (macro) | 0.9726 ± 0.0252 | 0.9484 |

- PCA 降至 2 维 (累积解释方差比: ~55.06%)
- SVM 核函数: RBF
