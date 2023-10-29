import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Select the first two features (sepal length and sepal width)
y = iris.target

# Create SVM Classifier with a linear kernel
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C, gamma='auto').fit(X, y)

# Create a mesh grid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min) / 100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for each point in the mesh grid
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and data points
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Classifier with Linear Kernel')
plt.show()
