import numpy as np
import matplotlib.pyplot as plt

# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

from sklearn import datasets,svm

# Load wine dataset
wine = datasets.load_wine()

# Get the 0th and 6th explanatory variables
X = wine.data[:, [0, 6]]

# Extract target (set target variable 0 to 1, 1, 2 to 0)
y = (wine.target == 0).astype(int)

# SVC model settings
clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)

# Train the model
clf.fit(X, y)

# Set graph size
plt.figure(figsize=(6, 4))

# Creating a scatter plot
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='class 0', marker='o', s=50)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Except for class 0', marker='^', s=50)

# Plot support vector
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, \
            facecolors='none', edgecolors='k', label='Support Vector')

# Plot the boundaries
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1.5
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # Margin calculation
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], \
            linestyles=['--', '-', '--'], levels=[-1, 0, 1])

# Set axis label
plt.xlabel('0th feature')
plt.ylabel('6th feature')

# Display graph
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
