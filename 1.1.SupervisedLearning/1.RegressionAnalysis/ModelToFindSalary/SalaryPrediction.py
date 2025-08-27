import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("salary_data.csv")

x = df.drop('Salary', axis=1) 
y = df['Salary']  

model = LinearRegression()

model.fit(x, y)

salaries = model.predict(pd.DataFrame([[15],[18]], columns=['Experience']))

print("Salary of 15 years of experience is:", salaries[0])
print("Salary of 18 years of experience is:", salaries[1])

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

