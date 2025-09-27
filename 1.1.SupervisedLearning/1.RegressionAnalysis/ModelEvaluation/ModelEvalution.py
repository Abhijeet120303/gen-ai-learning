import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("SpendingData.csv")
X = data.drop("Spendings", axis=1)
y = data["Spendings"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)

plt.scatter(X_train["Salary"], y_train, color='blue', label='Training data')
plt.scatter(X_test["Salary"], y_test, color='red', label='Testing data')
plt.xlabel("Salary")
plt.ylabel("Spendings")
plt.title("Train-Test Data Split")
plt.legend()
# plt.show()

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R^2 Score: {train_score}")
print(f"Test R^2 Score: {test_score}")

y_pred = model.predict(X_test)

# print("Predicted values:", y_pred)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error: {mape}%")