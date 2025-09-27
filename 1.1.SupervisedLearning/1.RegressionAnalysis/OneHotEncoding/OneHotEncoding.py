import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("salary_data.csv")

X = df.drop("Salary", axis=1)
y = df["Salary"]

column_transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=True, drop="first"), ["Title"])
    ],
    remainder="passthrough" 
)

transformed_values = column_transformer.fit_transform(X)

# print(transformed_values)

transformed_features = pd.DataFrame(
    transformed_values, 
    columns=column_transformer.get_feature_names_out()
)

# print(transformed_features)

model = LinearRegression()
model.fit(transformed_features, y)

data_to_predict = pd.DataFrame(
    [["Software Engineer", 1]],
    columns=["Title", "Experience"]
)

new_data_transformed = column_transformer.transform(data_to_predict)

data_frame_to_predict = pd.DataFrame(
    new_data_transformed,
    columns=column_transformer.get_feature_names_out()
)

y_pred = model.predict(data_frame_to_predict)
print(y_pred)
