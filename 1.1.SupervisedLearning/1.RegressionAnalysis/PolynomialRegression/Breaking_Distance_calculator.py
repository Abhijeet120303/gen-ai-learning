import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('Breking_Distance_Data.csv')

# df.info()
# df.describe()

corelation=df['Speed'].corr(df['BrakingDistance'])
print("Corelation between Speed and BrakingDistance is: ",corelation)

# plt.scatter(df['Speed'],df['BrakingDistance'])
# plt.xlabel('Speed')
# plt.ylabel('BrakingDistance')
# plt.title('Speed vs BrakingDistance')
# plt.show()

x=df.drop('BrakingDistance',axis=1)
y=df['BrakingDistance']

# print(x)

poly_features = PolynomialFeatures(degree=5)
x_poly = poly_features.fit_transform(x)

# print(x_poly)

model = LinearRegression()
model.fit(x_poly, y)

output= model.predict(poly_features.fit_transform([[330]]))
print(output)