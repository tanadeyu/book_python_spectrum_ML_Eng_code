import matplotlib.pyplot as plt
# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

from sklearn import datasets, model_selection, svm, metrics

# Data loading and splitting training and testing data
iris = datasets.load_iris()
X, y = iris.data[:, :], iris.target
X_train, X_test, y_train, y_test = \
model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

#jp_feature_names = \
#['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
#iris.feature_names = jp_feature_names
#jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']
#iris.target_names = jp_target_names

# Model selection and training
clf = svm.SVC(kernel='poly').fit(X_train, y_train)

#　Estimate calculation
y_pred = clf.predict(X)

# Calculating and displaying Accuracy
accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))

# Preparing to create a scatter plot (setting the axis)
expv = [2,3,0] # Select 3 explanatory variables
common_xlim = X[:, expv[0]].min()-0.5, X[:, expv[0]].max()+0.5
common_ylim = X[:, expv[1]].min()-0.5, X[:, expv[1]].max()+0.5
common_zlim = X[:, expv[2]].min()-0.5, X[:, expv[2]].max()+0.5

# Displaying a scatter plot
fig, (ax1, ax2) \
= plt.subplots(1, 2, subplot_kw={'projection':'3d'}, figsize=(10, 5))
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
for ax, x_data, y_data in zip([ax1, ax2], [X, X], [y, y_pred]):
    for i, color, marker in zip(clf.classes_, colors, markers):
        ax.scatter(x_data[y_data == i, expv[0]], x_data[y_data == i, expv[1]], \
        x_data[y_data == i, expv[2]], c=color, marker=marker, s=50,
        label=iris.target_names[i])
    ax.set_xlabel(iris.feature_names[expv[0]], fontsize=14)
    ax.set_ylabel(iris.feature_names[expv[1]], fontsize=14)
    ax.set_zlabel(iris.feature_names[expv[2]], fontsize=14)
    ax.set_xlim(common_xlim)
    ax.set_ylim(common_ylim)
    ax.set_zlim(common_zlim)
ax1.set_title("Original data")
ax2.set_title('Prediction data (Accuracy: {:.3f})'.format(accuracy), fontsize=14)
plt.legend(loc='best')
plt.show()