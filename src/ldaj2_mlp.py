import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 60)
print("LDA + MLP — MNIST 手写数字识别")
print("=" * 60)

mnist = fetch_openml('mnist_784', version=1)
x = mnist.data.values if hasattr(mnist.data, 'values') else mnist.data
y = mnist.target.astype(int).values if hasattr(mnist.target, 'values') else mnist.target.astype(int)

print(f"\n数据集大小: {x.shape[0]} 样本, {x.shape[1]} 特征")
print(f"类别数: {len(np.unique(y))}")

# ============================================================================
# 2. 划分测试集 (20% held-out，分层抽样)
# ============================================================================
x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练+验证集: {x_trainval.shape[0]} 样本")
print(f"测试集:      {x_test.shape[0]} 样本")

# ============================================================================
# 3. K 折交叉验证 (K=5)
# ============================================================================
K = 5
dimension = 9  # LDA 最大维度 = n_classes - 1 = 9

kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
cv_acc_scores = []
cv_precision_scores = []
cv_recall_scores = []
cv_f1_scores = []

print(f"\n--- {K} 折交叉验证 ---")

for fold, (train_idx, val_idx) in enumerate(kf.split(x_trainval, y_trainval)):
    x_train_fold = x_trainval[train_idx]
    x_val_fold = x_trainval[val_idx]
    y_train_fold = y_trainval[train_idx]
    y_val_fold = y_trainval[val_idx]

    # 3a. 归一化：仅用训练折拟合，再变换验证折 (防止数据泄漏)
    scaler = StandardScaler()
    x_train_fold_scaled = scaler.fit_transform(x_train_fold)
    x_val_fold_scaled = scaler.transform(x_val_fold)

    # 3b. LDA 降维
    lda = LinearDiscriminantAnalysis(n_components=dimension)
    x_train_fold_lda = lda.fit_transform(x_train_fold_scaled, y_train_fold)
    x_val_fold_lda = lda.transform(x_val_fold_scaled)

    # 3c. MLP 训练
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42,
    )
    mlp.fit(x_train_fold_lda, y_train_fold)

    # 3d. 验证
    y_val_pred = mlp.predict(x_val_fold_lda)
    cv_acc_scores.append(accuracy_score(y_val_fold, y_val_pred))
    cv_precision_scores.append(precision_score(y_val_fold, y_val_pred, average='macro', zero_division=0))
    cv_recall_scores.append(recall_score(y_val_fold, y_val_pred, average='macro', zero_division=0))
    cv_f1_scores.append(f1_score(y_val_fold, y_val_pred, average='macro', zero_division=0))

    print(f"  Fold {fold+1}: Accuracy={cv_acc_scores[-1]:.4f}, "
          f"Precision={cv_precision_scores[-1]:.4f}, "
          f"Recall={cv_recall_scores[-1]:.4f}, "
          f"F1={cv_f1_scores[-1]:.4f}")

print(f"\n交叉验证汇总:")
print(f"  Accuracy  = {np.mean(cv_acc_scores):.4f} ± {np.std(cv_acc_scores):.4f}")
print(f"  Precision = {np.mean(cv_precision_scores):.4f} ± {np.std(cv_precision_scores):.4f}")
print(f"  Recall    = {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}")
print(f"  F1-score  = {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")

# ============================================================================
# 4. 最终训练 (全量 trainval → 测试集评估)
# ============================================================================
print(f"\n--- 最终训练与测试 ---")

scaler = StandardScaler()
x_trainval_scaled = scaler.fit_transform(x_trainval)
x_test_scaled = scaler.transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=dimension)
x_trainval_lda = lda.fit_transform(x_trainval_scaled, y_trainval)
x_test_lda = lda.transform(x_test_scaled)

mlp_final = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='logistic',
    learning_rate_init=0.001,
    max_iter=50,
    random_state=42,
)
mlp_final.fit(x_trainval_lda, y_trainval)

y_pred = mlp_final.predict(x_test_lda)
test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\n测试集性能:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_precision:.4f} (macro)")
print(f"  Recall:    {test_recall:.4f} (macro)")
print(f"  F1-score:  {test_f1:.4f} (macro)")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================================
# 5. 可视化
# ============================================================================
print(f"\n--- 生成可视化 ---")
out_dir = "/home/yanwq/pattern_recognition/results/figures"

# ---- 5a. 混淆矩阵 ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=range(10), yticklabels=range(10))
plt.title("LDA + MLP — Confusion Matrix (Test Set)", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{out_dir}/lda_mlp_confusion_matrix.png", dpi=150)
plt.close()
print("  [1/3] 混淆矩阵已保存")

# ---- 5b. 决策面可视化 (前2个 LDA 分量) ----
# 用前 2 个 LDA 分量创建网格，其余 7 维用训练集均值填充
x_trainval_lda_2d = x_trainval_lda[:, :2]
x_test_lda_2d = x_test_lda[:, :2]
mean_remaining = x_trainval_lda[:, 2:].mean(axis=0)  # shape (7,)

x_min, x_max = x_trainval_lda_2d[:, 0].min() - 1, x_trainval_lda_2d[:, 0].max() + 1
y_min, y_max = x_trainval_lda_2d[:, 1].min() - 1, x_trainval_lda_2d[:, 1].max() + 1

# 控制网格密度，避免预测过慢
step = max((x_max - x_min) / 200, (y_max - y_min) / 200)
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

# 将 2D 网格点扩展为 9 维 (剩余维用训练集均值填充)
grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_full = np.hstack([grid_2d, np.tile(mean_remaining, (grid_2d.shape[0], 1))])
Z = mlp_final.predict(grid_full).reshape(xx.shape)

plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='tab10', levels=np.arange(-0.5, 10.5))
scatter = plt.scatter(x_test_lda_2d[:, 0], x_test_lda_2d[:, 1],
                      c=y_test, cmap='tab10', s=8, edgecolors='k', linewidth=0.3)
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.title("LDA + MLP — Decision Boundary (First 2 LDA Components)", fontsize=14)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.tight_layout()
plt.savefig(f"{out_dir}/lda_mlp_decision_boundary.png", dpi=150)
plt.close()
print("  [2/3] 决策面可视化已保存")

# ---- 5c. 综合指标图 ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

metrics_text = (
    f"LDA + MLP — Test Set Performance\n"
    f"{'─' * 35}\n\n"
    f"5-Fold Cross Validation (Train+Val):\n"
    f"  Accuracy  = {np.mean(cv_acc_scores):.4f} ± {np.std(cv_acc_scores):.4f}\n"
    f"  Precision = {np.mean(cv_precision_scores):.4f} ± {np.std(cv_precision_scores):.4f}\n"
    f"  Recall    = {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}\n"
    f"  F1-score  = {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}\n\n"
    f"Test Set (Hold-out 20%):\n"
    f"  Accuracy  = {test_acc:.4f}\n"
    f"  Precision = {test_precision:.4f} (macro avg)\n"
    f"  Recall    = {test_recall:.4f} (macro avg)\n"
    f"  F1-score  = {test_f1:.4f} (macro avg)\n\n"
    f"Configuration:\n"
    f"  LDA components = {dimension}\n"
    f"  MLP hidden     = (128, 64)\n"
    f"  Activation     = logistic\n"
    f"  Learning rate  = 0.001\n"
    f"  Max iterations = 50"
)
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.title("LDA + MLP — Performance Summary", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{out_dir}/lda_mlp_metrics_summary.png", dpi=150)
plt.close()
print("  [3/3] 性能指标汇总已保存")

print(f"\n所有结果已保存至 {out_dir}/")
print("=" * 60)
