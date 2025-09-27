import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

df=pd.read_csv('heart_disease.csv')

# print(df.info())

x = df.drop('target', axis=1)
y = df["target"]

# print(df.corr())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

output=model.predict(pd.DataFrame([[37,1,2,130,250,0,1,187,0,3.5,0,0,2]], columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']))

print(output)

y_pred=model.predict(x_test)

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :")
print(confusion_mat)

precison=precision_score(y_test, y_pred, average='weighted')
print("Precision Score")
print(precison)


accuracy=accuracy_score(y_test, y_pred)
print("Accuracy")
print(accuracy)

f1=f1_score(y_test,y_pred, average='weighted')
print("F1 Score :")
print(f1)

