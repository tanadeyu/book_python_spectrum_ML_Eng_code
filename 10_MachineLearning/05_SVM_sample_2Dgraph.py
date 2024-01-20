import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # Get the third and fourth explanatory variables
y = iris.target

# Perform standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Configure SVM model
classifiers = [
    SVC(kernel="linear",  C=3.0),
    SVC(kernel="poly",    degree=3, C=3.0),
    SVC(kernel="rbf",     gamma=0.5, C=3.0),
    SVC(kernel="sigmoid", gamma=0.5, C=3.0)
]

# Graph settings
fig, sub = plt.subplots(2, 2,figsize=(8, 6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Learning and drawing with each model (clf is classification)
for clf, title, ax  in zip(classifiers, \
    ['Linear SVM', 'Poly SVM', 'RBF SVM', 'Sigmoid SVM'], sub.flatten()):
    clf.fit(X_train, y_train)
    # Drawing a scatter plot
    for i, color, marker in zip(clf.classes_, ['r', 'g', 'b'], \
        ['o', '^', ',']):
        idx = np.where(y_train == i)
        ax.scatter(X_train[idx, 0], X_train[idx, 1], c=color, \
        label=iris.target_names[i], \
        marker=marker, edgecolor='k', s=60)
    # Drawing border
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), \
    np.linspace(ylim[0], ylim[1], 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pltt=ax.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.binary)
    ax.set_xlabel('Feature 3')
    ax.set_ylabel('Feature 4')
    ax.set_title(title)
    fig.colorbar(pltt,ax=ax)

    # Show support vectors with circles
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], \
    s=300, facecolors='none', edgecolors='red')

    # Calculating and displaying accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ax.text(0.95, 0.05, ('Accuracy = %.2f' % acc).lstrip('0'), \
    size=12, ha='right', transform=ax.transAxes, color='black')

plt.tight_layout()
plt.show()