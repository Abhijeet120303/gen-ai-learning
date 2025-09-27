import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
df = pd.read_csv("salary_data.csv")

# Step 2: Separate rows where Salary is null
null_data = df[df['Salary'].isnull()]
non_null_data = df[df['Salary'].notnull()]

# Step 3: Training data (Experience → Salary)
X = non_null_data[['Experience']]
y = non_null_data['Salary']

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

print("Model trained on non-null data")

# Step 5: Predict Salary for missing rows
if not null_data.empty:
    X_null = null_data[['Experience']]
    predicted_salaries = model.predict(X_null)
    
    # Fill back into DataFrame
    df.loc[df['Salary'].isnull(), 'Salary'] = predicted_salaries

# Step 6: Save updated CSV
df.to_csv("salary_data_filled.csv", index=False)
print("Missing Salary values filled and saved into salary_data_filled.csv")

# Step 7: Train final model on complete dataset
X_full = df[['Experience']]
y_full = df['Salary']
final_model = LinearRegression()
final_model.fit(X_full, y_full)

print("Final model trained on full data!")
