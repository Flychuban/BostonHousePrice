import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle

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

pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Model Training
regression = LinearRegression()
param_grid = {'fit_intercept': [True, False], 'copy_X': [True, False], "positive": [True, False]}
grid = GridSearchCV(regression, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

grid.best_params_
print("Best score ", np.sqrt(-grid.best_score_))

# Model Evaluation on training data
y_train_pred = grid.predict(X_train)
print("RMSE: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("MAE: ", mean_absolute_error(y_train, y_train_pred))
print("R2 Score: ", r2_score(y_train, y_train_pred))

# Model Evaluation on test data
y_test_pred = grid.predict(X_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("MAE: ", mean_absolute_error(y_test, y_test_pred))
print("R2 Score: ", r2_score(y_test, y_test_pred))


# New data predictions
grid.predict(scaler.transform(df.iloc[0][:13].values.reshape(1, -1)))

filename = 'boston_housing_model.sav'
pickle.dump(grid, open(filename, 'wb'))