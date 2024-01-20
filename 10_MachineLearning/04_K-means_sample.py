import os
os.environ["OMP_NUM_THREADS"] = "1"
value = os.environ["OMP_NUM_THREADS"]
print(value)
#import warnings
#warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

#import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data

# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Japanese support for feature_names and target_names
#jp_feature_names = \
#    ['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
#iris.feature_names = np.array(jp_feature_names)
#jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']
#iris.target_names = np.array(jp_target_names)

# Drawing clustering results 1
exp_var1, exp_var2 = 0, 1 # Use explanatory variables 0 and 1
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
markers = ['o', '^', 's']
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X[y_kmeans == i, exp_var1], X[y_kmeans == i, exp_var2], \
    s=50, c=colors[i], marker=markers[i], label=f'Cluster {i}')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, exp_var1], centers[:, exp_var2], \
s=300, c='black', marker='*', label='Centroids') # Display Centroids

# Display graph
plt.legend()
plt.xlabel(iris.feature_names[exp_var1])
plt.ylabel(iris.feature_names[exp_var2])
plt.tight_layout()
ax.axis('equal')
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
plt.show()

# Drawing clustering results 2
exp_var1, exp_var2 = 2, 3 # Use explanatory variables 2 and 3
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
markers = ['o', '^', 's']
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X[y_kmeans == i, exp_var1], X[y_kmeans == i, exp_var2], \
    s=50, c=colors[i], marker=markers[i], label=f'Cluster {i}')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, exp_var1], centers[:, exp_var2], \
s=300, c='black', marker='*', label='Centroids') # Display Centroids

# Display graph
plt.legend()
plt.xlabel(iris.feature_names[exp_var1])
plt.ylabel(iris.feature_names[exp_var2])
plt.tight_layout()
ax.axis('equal')
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
plt.show()
