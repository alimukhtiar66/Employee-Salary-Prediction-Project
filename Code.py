import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


Dataset = pd.read_csv('Dataset.csv')

X = Dataset.iloc[:,[0,1,2,3,4,5,6]]
Y = Dataset.iloc[:,7]


X_train, X_test, y_train, y_test = train_test_split(X, Y,  random_state=0)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

