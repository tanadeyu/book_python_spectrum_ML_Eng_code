import matplotlib.pyplot as plt
# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'

import numpy as np
from scipy import stats
from sklearn.datasets import load_wine

# Load wine dataset
data = load_wine()
X = data.data
#fnames_jp = \
#["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
# "マグネシウム", "全フェノール含量", "フラボノイド", \
# "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
# "色相", "OD280/OD315", "プロリン"]
#data.feature_names = fnames_jp


# 13 subplot settings for explanatory variables
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
axs = axs.ravel()

# Create a histogram for each feature
for i in range(X.shape[1]):
    ax = axs[i]
    feature = X[:, i]
    ax.hist(feature, bins=15, density=True, alpha=0.9, color='w', histtype="bar",edgecolor="black")

    # Plot normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(feature), np.std(feature))
    ax.plot(x, p, 'k', linewidth=2)

    # Shapiro-Wilk test p-value and statistics
    stat, p_value = stats.shapiro(feature)
    ax.text(0.05, 0.8, f'P-value: {p_value:.3f}\nStatistic: {stat:.3f}', transform=ax.transAxes)
    ax.set_title(data.feature_names[i])

plt.tight_layout()
plt.show()
