import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler

dataset_feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"] 
dataset = pd.read_csv('data/housing.csv', header=None, delimiter=r"\s+", names=dataset_feature_names)
df = pd.DataFrame(dataset, columns=dataset_feature_names)

# Data Analysis
df.info()
df.describe()

# Check for missing values
df.isnull().sum()

plt.scatter(df["CRIM"], df["MEDV"])
plt.xlabel("Per capita crime rate by town")
plt.ylabel("Median value of owner-occupied homes in $1000's")

sns.regplot(x="RM", y="MEDV", data=df)

# Correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr().abs(),  annot=True)

# Dependant and independant variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)