import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 60)
print("PCA + SVM — Wine 葡萄酒分类")
print("=" * 60)

wine = load_wine()
x = wine.data
y = wine.target

print(f"\n数据集大小: {x.shape[0]} 样本, {x.shape[1]} 特征")
print(f"类别数: {len(np.unique(y))}  ({list(wine.target_names)})")

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
dimension = 2  # PCA 降维到 2 维

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

    # 3b. PCA 降维
    pca = PCA(n_components=dimension)
    x_train_fold_pca = pca.fit_transform(x_train_fold_scaled)
    x_val_fold_pca = pca.transform(x_val_fold_scaled)

    # 3c. SVM 训练
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(x_train_fold_pca, y_train_fold)

    # 3d. 验证
    y_val_pred = svm.predict(x_val_fold_pca)
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

pca = PCA(n_components=dimension)
x_trainval_pca = pca.fit_transform(x_trainval_scaled)
x_test_pca = pca.transform(x_test_scaled)

print(f"PCA 解释方差比: {pca.explained_variance_ratio_}  (合计: {pca.explained_variance_ratio_.sum():.4f})")

svm_final = SVC(kernel='rbf', random_state=42)
svm_final.fit(x_trainval_pca, y_trainval)

y_pred = svm_final.predict(x_test_pca)
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
print(classification_report(y_test, y_pred, target_names=wine.target_names, zero_division=0))

# ============================================================================
# 5. 可视化
# ============================================================================
print(f"\n--- 生成可视化 ---")
out_dir = "/home/yanwq/pattern_recognition/results/figures"

# ---- 5a. 混淆矩阵 ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("PCA + SVM — Confusion Matrix (Test Set)", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{out_dir}/pca_svm_confusion_matrix.png", dpi=150)
plt.close()
print("  [1/4] 混淆矩阵已保存")

# ---- 5b. PCA 散点图 ----
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_test,
                      cmap='viridis', s=50, edgecolors='k')
plt.title("PCA — Test Set Projection", fontsize=14)
plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.colorbar(scatter, ticks=range(3), label='Class')
plt.tight_layout()
plt.savefig(f"{out_dir}/pca_svm_pca_projection.png", dpi=150)
plt.close()
print("  [2/4] PCA 投影图已保存")

# ---- 5c. SVM 决策面 ----
x_min = x_trainval_pca[:, 0].min() - 0.5
x_max = x_trainval_pca[:, 0].max() + 0.5
y_min = x_trainval_pca[:, 1].min() - 0.5
y_max = x_trainval_pca[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_final.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_test,
                      cmap='viridis', s=50, edgecolors='k', linewidth=0.5)
plt.title("PCA + SVM — Decision Boundary (Test Set)", fontsize=14)
plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.colorbar(scatter, ticks=range(3), label='Class')
plt.tight_layout()
plt.savefig(f"{out_dir}/pca_svm_decision_boundary.png", dpi=150)
plt.close()
print("  [3/4] SVM 决策面已保存")

# ---- 5d. 综合指标图 ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

metrics_text = (
    f"PCA + SVM — Test Set Performance\n"
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
    f"  PCA components = {dimension}\n"
    f"  Variance retained = {pca.explained_variance_ratio_.sum():.4f}\n"
    f"  SVM kernel  = RBF\n"
)
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.title("PCA + SVM — Performance Summary", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{out_dir}/pca_svm_metrics_summary.png", dpi=150)
plt.close()
print("  [4/4] 性能指标汇总已保存")

print(f"\n所有结果已保存至 {out_dir}/")
print("=" * 60)
