import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv("Advertising.csv")

# df.info()
# print(df.columns)
# df.describe()

# plt.scatter(df['TV'],df['sales'])
# plt.xlabel('TV')
# plt.ylabel('sales')
# plt.title('TV vs sales')
# plt.show()

# plt.scatter(df['radio'],df['sales'])
# plt.xlabel('radio')
# plt.ylabel('sales')
# plt.title('radio vs sales')
# plt.show()

# plt.scatter(df['newspaper'],df['sales'])
# plt.xlabel('newspaper')
# plt.ylabel('sales')
# plt.title('newspaper vs sales')
# plt.show()

# corelation=df['TV'].corr(df['sales'])
# print("Corelation between TV and sales is: ",corelation)

# corelation=df['radio'].corr(df['sales'])
# print("Corelation between radio and sales is: ",corelation)

# corelation=df['newspaper'].corr(df['sales'])
# print("Corelation between newspaper and sales is: ",corelation)


x=df.drop('sales', axis=1)
y=df['sales']

model=LinearRegression()
model.fit(x,y)

sales = model.predict(pd.DataFrame([[180,40,32]],columns=['TV','radio','newspaper']))
print(sales[0])