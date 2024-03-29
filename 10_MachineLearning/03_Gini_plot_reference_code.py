import matplotlib.pyplot as plt

# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np

# Loading data
iris = load_iris()
X = iris.data[:, 2:]  # Use only petal length and width
y = iris.target

# Feature_name label
# jp_feature_names = \
#    ['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
# iris.feature_names = jp_feature_names

# Target_name label
# jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']
en_target_names = ['Setosa', 'Versicolor', 'Virginica']
target_names = en_target_names

# Creation of a decision tree model
tree_clf = DecisionTreeClassifier(max_depth=3, criterion="gini",random_state=10)
tree_clf.fit(X, y)

# Calculation of Gini impurity value (for display)
def gini(p):
    return 1 - (p ** 2).sum(axis=1)

# Plot by class value (for border visualization)
fig = plt.figure(figsize=(8, 5))
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.1, cmap="binary")
plt.colorbar(format="%.0f", label='Class',ticks=[0,1,2])

# Overlay scatter plots for each class
for i, marker, color in zip(range(3), ['o', '^', ','], ['r', 'g', 'b']):
    plt.scatter(X[y == i, 0], X[y == i, 1], marker=marker, s=40, c=color, \
                label=target_names[i],facecolor='None')
plt.xlabel(iris.feature_names[2], fontsize=12)
plt.ylabel(iris.feature_names[3], fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
plt.legend(loc='best', fontsize=12)
plt.show()

# Overwriting Gini impurity values (for class contourf)
gini_values = gini(tree_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]))
gini_values = gini_values.reshape(xx.shape)
plt.contourf(xx, yy, gini_values, alpha=0.1, cmap="gist_ncar")
plt.colorbar(format="%.2f", label='Gini Impurity')
plt.show()

# Visualization of decision trees
plt.figure(figsize=(10,5))
plot_tree(tree_clf, feature_names=iris.feature_names[2:], \
          class_names=target_names, filled=False, fontsize=11)
plt.tight_layout()
plt.show()

