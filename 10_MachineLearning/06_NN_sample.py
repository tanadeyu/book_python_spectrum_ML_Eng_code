import matplotlib.pyplot as plt
# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Yu Gothic'

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load wine dataset
data = load_wine()
X = data.data
y = data.target

# feature_names_jp = \
# ["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
#  "マグネシウム", "全フェノール含量", "フラボノイド", \
#  "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
#  "色相", "OD280/OD315", "プロリン"]
#data.feature_names = feature_names_jp

# Select 0th and 6th features
X = X[:, [0, 6]]

# Data normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Separate into training data and test data
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.4, random_state=0)

# Neural network model settings
model = MLPClassifier(hidden_layer_sizes=(10, 10), \
                      activation='relu', solver='sgd', \
                      max_iter=5000, random_state=0,tol=1e-5)

# Train the model
history = model.fit(X_train, y_train)

# Prediction of test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Figure size settings
plt.figure(figsize=(8, 4))

# Display a scatter plot of the original data
plt.subplot(1, 2, 1)
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
for i in range(3):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], \
    c=colors[i], marker=markers[i], label=f'Class {i}')
plt.title('Original data')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[6])
plt.legend(loc='upper right')

# Display a scatter plot of the prediction Data
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], \
    c=colors[i], marker=markers[i], label=f'Class {i}')
plt.title('Prediction Data')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[6])
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

# Display the number of trials and loss value
plt.figure(figsize=(4, 4))
plt.plot(history.loss_curve_)
plt.title('Loss curve')
plt.xlabel('Number of trials')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()