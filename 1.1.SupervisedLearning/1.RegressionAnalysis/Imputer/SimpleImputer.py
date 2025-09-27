import pandas as pd 
from sklearn.impute import SimpleImputer

df=pd.read_csv('salary_data.csv')

# mean, median, constant, most_frequent

imputer=SimpleImputer(strategy='mean')
df[['Salary']] = imputer.fit_transform(df[['Salary']])

print(df)


