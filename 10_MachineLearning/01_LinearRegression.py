import pandas as pd
import matplotlib.pyplot as plt

# Fonts setting
plt.rcParams['font.family'] = 'PT Serif'
#plt.rcParams['font.family'] = 'Times New Roman'
# Japanese Fonts
#plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
# Loading the California House Price Dataset
# X is the explanatory variable, y is the target variable
california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target
feature_names = california_housing.feature_names
 
# Combine X and y data once (convert to Pandas DataFrame)
data = pd.DataFrame(data=X, columns=feature_names)
data['Target'] = y
 
# Display some data contents
print("Data:")
print(data.head())
 
# Divide into explanatory variables and target variables again
# also possible to specify a target variable other than Target
X = data.drop('Target', axis=1)
y = data['Target']
 
# Split the data (80% for training, 20% for testing)
# Specify random_state to check reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, \
  test_size=0.2, random_state=42)
 
# Model selection and training
model = LinearRegression()
model.fit(X_train, y_train)
 
# Prediction on test data
y_pred = model.predict(X_test)
 
# Model evaluation (mean squared error)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

# Intercept and coefficient
print("Bias:", model.intercept_)
print("Coef:", model.coef_)
 
# Plot predicted and actual values
fig = plt.figure(figsize=(4, 7))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, y_pred, color='blue',s=3)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual price and Predicted price")
ax1.set_aspect('equal')
#plt.tight_layout()
plt.show()

# Display of weights
#feature_names_jp = ['世帯所得', '築年数', '部屋数平均', \
#                    '寝室数平均', '居住人数', '世帯人数平均', \
#                    '代表地区緯度', '代表地区経度']
fig2 = plt.figure(figsize=(6, 3))
coeff = pd.Series(model.coef_, index=california_housing.feature_names)
#coeff = pd.Series(model.coef_, index=feature_names_jp)
coeff.plot(kind='bar')
plt.tight_layout()
plt.show()
