import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Read the model from the file
with open('braking_distance_model.pkl', 'rb') as file:
    model = pickle.load(file)

poly=PolynomialFeatures(degree=5)
data_to_predict = poly.fit_transform([[120]])

print(data_to_predict)
output=model.predict(data_to_predict)

print("120",output)
