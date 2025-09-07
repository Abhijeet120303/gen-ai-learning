import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('crop_yield_dataset.csv')

# df.info()

# print(df.describe())

corelation=df['fertilizer_kg_per_acre'].corr(df['crop_yield_quintals'])
print("Corelation between fertilizer_kg_per_acre and crop_yield_quintals is: ",corelation)

# plt.scatter(df['fertilizer_kg_per_acre'],df['crop_yield_quintals'])
# plt.xlabel('fertilizer_kg_per_acre')
# plt.ylabel('crop_yield_quintals')
# plt.title('fertilizer_kg_per_acre vs crop_yield_quintals')
# plt.show()

x=df.drop('crop_yield_quintals',axis=1)
y=df['crop_yield_quintals']

poly_features=PolynomialFeatures(degree=4)
x_poly=poly_features.fit_transform(x)

model=LinearRegression()
model.fit(x_poly,y)

output= model.predict(poly_features.fit_transform([[100]]))
print(output)