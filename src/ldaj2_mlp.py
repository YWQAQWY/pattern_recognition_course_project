from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)

x = mnist.data
y = mnist.target.astype(int)

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

dimension = 9
lda = LinearDiscriminantAnalysis(n_components=dimension)
x_train_lda = lda.fit_transform(x_train, y_train)
x_val_lda = lda.transform(x_val)
x_test_lda = lda.transform(x_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(128,64),
    activation='sigmoid',
    learning_rate= 0.001,
    max_iter=50,
    random_state=42
)

mlp.fit(x_train_lda, y_train)

y_pred = mlp.predict(x_test_lda)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# Visualization
plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)
plt.title("LDA + MLP Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("/home/yanwq/pattern_recognition/results/figures/mnist_lda_mlp_confusion_matrix.png")
plt.close()

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    x_test_lda[:,0],
    x_test_lda[:,1],
    c=y_test,
    s=10
)
plt.title("MNIST LDA Visualization")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(scatter)
plt.savefig("/home/yanwq/pattern_recognition/results/figures/mnist_lda_visualization.png")
plt.close()

#========================================================================
# VALIDATION
#========================================================================
y_val_pred = mlp.predict(x_val_lda)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")


