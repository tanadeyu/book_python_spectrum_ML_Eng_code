import matplotlib.pyplot as plt

# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'

# Import the required libraries
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Wine dataset
wine = load_wine()

# Convert to dataframe
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Replace data items with Japanese
#feature_names_jp = \
#["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
# "マグネシウム", "全フェノール含量", "フラボノイド", \
# "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
# "色相", "OD280/OD315", "プロリン"]
#df.columns = feature_names_jp

# Split the data into training and testing
X_train, X_test, y_train, y_test \
= train_test_split(df, wine.target, test_size=0.3, random_state=0)

# Normalization (improve model accuracy of logistic regression)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# logistic regression model (C is the initial value of 1.0)
lr = LogisticRegression(C=1.0, random_state=0)
lr.fit(X_train_std, y_train)

# Make predictions on test data and calculate accuracy rate
y_pred = lr.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % accuracy)

# Display learned parameters
print(lr.coef_)  # coefficient
print(lr.intercept_)  # intercept

# Display using graphs and labels
plt.figure(figsize=(5, 4))
colors = ['r', 'g', 'b']
markers = ['o', '^', ","]
lw = 2

# With the explanatory variable 0 alcohol content and 6 flavonoids
# Graph wine classes in a scatter plot
for mtemp, ctmp, i, target_name \
in zip(markers, colors, [0, 1, 2], wine.target_names):
    plt.scatter(wine.data[wine.target == i, 0], \
    wine.data[wine.target == i, 6], \
    marker=mtemp, color=ctmp, alpha=.8, lw=lw, \
    label=target_name)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[6])
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Wine dataset')
plt.show()

