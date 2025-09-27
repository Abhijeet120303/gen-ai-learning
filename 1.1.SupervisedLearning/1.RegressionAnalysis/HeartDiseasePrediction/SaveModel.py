import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('heart_disease.csv')

x = df.drop('target', axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

with open('HeartDiseaseModel.pkl', 'wb') as file:
    pickle.dump(model, file)
    
print("Model Saved Successfully")

