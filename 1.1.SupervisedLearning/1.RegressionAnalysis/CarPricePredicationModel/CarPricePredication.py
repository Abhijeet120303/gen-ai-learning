import pandas as pd
from sklearn.linear_model import LinearRegression


df=pd.read_csv("car_price_data.csv")

# .................. Data Analysis .................     

# print(df.columns)
# df.info()

# corelation=df['Age'].corr(df['Price'])
# print("Corelation between Age and Price is: ",corelation)

# corelation1=df['Mileage'].corr(df['Price'])
# print("Corelation between Mileage and Price is: ",corelation1)

x = df.drop('Price', axis=1)
y=df['Price']

model=LinearRegression()
model.fit(x,y)

price = model.predict(pd.DataFrame([[9,11000]],columns=['Age','Mileage']))
print(price)

# price = model.predict([[2,20000]])
# print(price)


