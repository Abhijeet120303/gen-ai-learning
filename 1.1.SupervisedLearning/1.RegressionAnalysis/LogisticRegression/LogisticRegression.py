import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

df=pd.read_csv('hearing_test.csv')

# print(df.info())
# print(df.corr())

# hearing_problem=df[df['test_result']==1]
# n0_hearing_problem=df[df['test_result']==0]

# plt.scatter(hearing_problem['age'],hearing_problem['physical_score'],color='red',marker='.')
# plt.scatter(n0_hearing_problem['age'],n0_hearing_problem['physical_score'],color='blue',marker='.')
# plt.show()

x=df.drop('test_result', axis=1)
y=df['test_result']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

output=model.predict(pd.DataFrame([[5,35 ]], columns=['age','physical_score']))
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

