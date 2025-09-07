import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pickle

df=pd.read_csv('Breking_Distance_Data.csv')

corr=df['Speed'].corr(df['BrakingDistance'])
print("Corelation between Speed and BrakingDistance is: ",corr)

x=df.drop('BrakingDistance',axis=1)
y=df['BrakingDistance']

poly_features = PolynomialFeatures(degree=5)
x_poly = poly_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

# save model in to the File
with open('braking_distance_model.pkl', 'wb') as file:
    pickle.dump(model, file)

output=model.predict(poly_features.fit_transform([[120]]))
print("Predicted Braking Distance for Speed 330 is: ", output)
