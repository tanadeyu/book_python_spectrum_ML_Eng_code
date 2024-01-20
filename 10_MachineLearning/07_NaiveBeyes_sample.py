import matplotlib.pyplot as plt
# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np
from scipy import stats

# Load wine dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target
#feature_names_jp = \
#["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
# "マグネシウム", "全フェノール含量", "フラボノイド", \
# "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
# "色相", "OD280/OD315", "プロリン"]
##wine.feature_names=feature_names_jp

# Split the data into training and testing
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.5, random_state=42)

# Create and fit a Gaussian-naive Bayesian model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Show model accuracy
print("accuracy:", gnb.score(X_test, y_test))
score_txt = "(accuracy: " + '{:.4f}'.format(gnb.score(X_test, y_test)) + ')'

# Calculate predicted value
predicted = gnb.predict(X_test)

# Create a scatter plot with original data and prediction
# (use index of 0 and 1 for explanatory variables)
fig, ax = plt.subplots(1, 2, figsize=(7, 4))
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
titles = ['Original data', 'Prediction'+score_txt]
dsets = [y_test, predicted]
for i in range(2):
    for j, color in enumerate(colors):
        ax[i].scatter(X_test[dsets[i] == j, 0], X_test[dsets[i] == j, 1], \
        marker=markers[j], label=wine.target_names[j], s=50,\
        facecolor='None', edgecolors=color)
        ax[i].set_xlabel(wine.feature_names[0])
        ax[i].set_ylabel(wine.feature_names[1])
        ax[i].legend(loc='best')
        ax[i].set_title(titles[i])
plt.tight_layout()
plt.show()

# Probability density diagram
fig, ax = plt.subplots(2, 3, figsize=(9, 5))
for i in range(2):
  for j in range(3):
    x = np.linspace(X_test[:, i].min(), X_test[:, i].max(), 100)
    y = stats.norm.pdf(x, gnb.theta_[j, i], np.sqrt(gnb.var_[j, i]))
    #area = np.trapz(y, x);print(area)
    #y_normalized = y / area
    ax[i][j].plot(x, y)
    ax[i][j].set_title(wine.target_names[j])
    ax[i][j].set_xlabel(wine.feature_names[i])
    ax[i][j].set_ylabel('Probability density')
plt.tight_layout()
plt.show()
