import matplotlib.pyplot as plt

# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

import pandas as pd
#import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Japanese feature_names
#jp_feature_names = \
#    ['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
#jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']
#iris.feature_names = jp_target_names

# Displaying a scatter plot
markers = ['o', '^', ","]
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
  for j in range(4):
    if i != j:
      for target in range(3):
        axs[i, j].scatter(iris.data[iris.target == target, i],\
                          iris.data[iris.target == target, j],\
                          label=iris.target_names[target],\
                          marker=markers[target])
      axs[i, j].set_xlabel(iris.feature_names[i],fontsize=12)
      axs[i, j].set_ylabel(iris.feature_names[j],fontsize=12)
      axs[i, j].legend(fontsize=10).get_frame().set_alpha(0.45)
plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# Create a decision tree model
dt_model = DecisionTreeClassifier(max_depth=2,random_state=0)
dt_model.fit(X_train, y_train)

# Create a random forest model
rf_model = RandomForestClassifier(n_estimators=10, \
                                  max_depth=2,random_state=0)
rf_model.fit(X_train, y_train)

# Output model accuracy
print("Decision Tree Accuracy:",   dt_model.score(X_test, y_test))
print("Random Forest Accuracy:",   rf_model.score(X_test, y_test))

# Importance of features (Decision tree)
feature_importances = dt_model.feature_importances_
feature_names = iris.feature_names
fig = plt.figure(figsize=(6, 3))
coeff = pd.Series(feature_importances, index=feature_names)
coeff.plot(kind='bar',fontsize=12)
plt.title("Importance of features (Decision tree)")
plt.tight_layout()
plt.show()

# Importance of features (Random forest)
feature_importances = rf_model.feature_importances_
feature_names = iris.feature_names
fig = plt.figure(figsize=(6, 3))
coeff = pd.Series(feature_importances, index=feature_names)
coeff.plot(kind='bar',fontsize=12)
plt.title("Importance of features (Random forest)")
plt.tight_layout()
plt.show()

# Visualization of decision trees
plt.figure(figsize=(10,5))
plot_tree(dt_model, feature_names=iris.feature_names, \
  class_names=iris.target_names.tolist(), filled=False, fontsize=11)
plt.tight_layout()
plt.show()

#　Visualization of random forest [0]
plt.figure(figsize=(10,5))
plot_tree(rf_model.estimators_[0], feature_names=iris.feature_names, \
  class_names=iris.target_names.tolist(), filled=False, fontsize=11)
plt.tight_layout()
plt.show()

#　Visualization of random forest [1]
plt.figure(figsize=(10,5))
plot_tree(rf_model.estimators_[1], feature_names=iris.feature_names, \
  class_names=iris.target_names.tolist(), filled=False, fontsize=11)
plt.tight_layout()
plt.show()

