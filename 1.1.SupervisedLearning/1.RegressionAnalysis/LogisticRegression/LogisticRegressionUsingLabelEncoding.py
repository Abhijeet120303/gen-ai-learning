import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

df=pd.read_csv('data.csv')

# print(df.info())

encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])

# print(df["species"])

x = df.drop('species', axis=1)
y = df["species"]

# print(df.corr())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

output=model.predict(pd.DataFrame([[5.7,2.8,4.5,1.3]], columns=['sepal_length','sepal_width','petal_length','petal_width']))

decoded= encoder.inverse_transform(output)

print(decoded)

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

