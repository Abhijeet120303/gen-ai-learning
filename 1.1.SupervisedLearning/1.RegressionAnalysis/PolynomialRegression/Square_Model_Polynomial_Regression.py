import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df=pd.read_csv('data.csv')

# plt.scatter(df['X'],df['Y'])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Scatter Plot of X vs Y')
# plt.show()

x=df.drop('Y',axis=1)
y=df['Y']

# print(x)

poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x)

# print(x_poly)

model = LinearRegression()
model.fit(x_poly, y)

output=model.predict(poly_features.fit_transform([[655]]))
print(output)