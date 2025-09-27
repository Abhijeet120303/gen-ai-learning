import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv')

X = df.drop('Salary', axis=1)
y = df['Salary']

encoder = LabelEncoder()
X['Title'] = encoder.fit_transform(X['Title'])

model = LinearRegression()
model.fit(X, y)

output = model.predict(pd.DataFrame([[1, 8]], columns=['Title', 'Experience']))

print(output)