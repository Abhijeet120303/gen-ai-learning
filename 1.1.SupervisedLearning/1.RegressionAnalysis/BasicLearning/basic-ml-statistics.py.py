# pandas help to handle data in tabular form.
import pandas as pd

# matplotlib help to visualize the data.
import matplotlib.pyplot as plt

# numpy help to perform mathematical operations on data.
import numpy as np

df=pd.read_csv('salary_data.csv')

#print("############# Columns : ")
print(df.columns)

#print("############# Info : ")
df.info()

#print("############# Describe : ")
print(df.describe())

# Plot scatter graph
# Visualize the data
# A scatter plot is a graph that shows the relationship between two variables
# The x-axis represents one variable, and the y-axis represents the other
# Each point on the graph represents a single observation
# Scatter plots are used to see if there is a relationship between two variables
# Scatter plots are used to see if there is a correlation between two variables
# Scatter plots are used to see if there is a pattern between two variables
# Scatter plots are used to see if there is a trend between two variables

plt.scatter(df['Experience'],df['Salary'])
plt.xlabel('Experiencence')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()

# Find correlation (Relation is stron ok)
# Correlation is a normalized form of covariance
# It is a measure of how much two variables change together
# Correlation ranges from -1 to 1
# 1 means that the variables are perfectly correlated
# 0 means that the variables are not correlated
# -1 means that the variables are perfectly inversely correlated

corelation=df['Experience'].corr(df['Salary'])
print("Corelation between Experience and Salary is: ",corelation)

# Find covariance (Find is there any relation between two variables)
# Covariance is a measure of how much two variables change together
# Positive covariance means that the variables are directly proportional
# Negative covariance means that the variables are inversely proportional
# Zero covariance means that the variables are not related
# Covariance can be any value
# Covariance is not normalized,standardized,bounded,percentage,probability,correlation
## Covariance of x and y = Σ((x - mean(x)) * (y - mean(y))) / (n-1)
## Covariance of x = Covariance of y
##[[cov(x,y)]] = cov(x,x) = cov(y,y)

covariance = np.cov(df['Experience'], df['Salary'])
print("Covariance: ", covariance)


# Mean, Median, Mode
# Mean is the average of all values
# Median is the middle value when all values are sorted
# Mode is the most frequently occurring value
# Mean, Median, Mode are used to understand the distribution of data

print('Mean of Experience: ', df['Experience'].mean())
print('Mean of Salary: ', df['Salary'].mean())
print('Median of Experience: ', df['Experience'].median())
print('Median of Salary: ', df['Salary'].median())
print('Mode of Experience: ', df['Experience'].mode()[0])
print('Mode of Salary: ', df['Salary'].mode()[0])
print('Standard Deviation of Experience: ', df['Experience'].std())
print('Standard Deviation of Salary: ', df['Salary'].std())
print('Variance of Experience: ', df['Experience'].var())                 
print('Variance of Salary: ', df['Salary'].var())