import numpy as np
import matplotlib.pyplot as plt

# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

from sklearn import datasets
from sklearn import svm

# Load Iris dataset
iris = datasets.load_iris()

# Omit class 2 and use the 3rd and 4th explanatory variables
X = iris.data[iris.target!=2]
X = X[:, 2:4]
# Set target variable other than 2
y = iris.target[iris.target!=2]

# Create SVC model
clf = svm.SVC(kernel='linear')

# Training the model
clf.fit(X, y)

# Set figure size
plt.figure(figsize=(6, 4))

# Draw scatter plot
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Target 0', marker='o',s=50)
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Target 1', marker='^',s=50)

# Plot support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, \
            facecolors='none', edgecolors='k', label='Support Vector')

# Show margins and their values
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
plt.text(2.5, 2, f'Margin value: {margin:.2f}', fontsize=12)

# Show borders
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], \
            linestyles=['--', '-', '--'], levels=[-margin, 0, margin])

# Set axis label
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

# Show graph
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()