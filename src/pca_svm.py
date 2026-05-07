from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from seaborn import heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

wine = load_wine()

x = wine.data
y = wine.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(
    x_scaled,
    y,
    test_size=0.3,
    random_state=42,
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

dimension = 2
pca = PCA(n_components= dimension)
x_train_pca = pca.fit_transform(x_train)
x_val_pca = pca.transform(x_val)
x_test_pca = pca.transform(x_test)

svm= SVC(kernel='rbf')

svm.fit(x_train_pca, y_train)

y_pred = svm.predict(x_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("PCA + SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("/home/yanwq/pattern_recognition/results/figures/confusion_matrix.png")
plt.close()

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    x_test_pca[:, 0],
    x_test_pca[:, 1],
    c=y_test
)
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(*scatter.legend_elements(), title="Classes")
plt.savefig("/home/yanwq/pattern_recognition/results/figures/pca_visualization.png")
plt.close()

"""SVM Visualization"""
# 生成二维平面范围
x_min = x_train_pca[:, 0].min() - 1
x_max = x_train_pca[:, 0].max() + 1

y_min = x_train_pca[:, 1].min() - 1
y_max = x_train_pca[:, 1].max() + 1


# 生成网格点
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)


# SVM预测整个平面
Z = svm.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

# 变回网格形状
Z = Z.reshape(xx.shape)


# 绘图
plt.figure(figsize=(8, 6))

# 分类区域
plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

# 样本点
scatter = plt.scatter(
    x_test_pca[:, 0],
    x_test_pca[:, 1],
    c=y_test,
    edgecolors='k'
)

plt.title("SVM Decision Boundary")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend(*scatter.legend_elements(), title="Classes")

plt.savefig("/home/yanwq/pattern_recognition/results/figures/svm_decision_boundary.png")

plt.close()