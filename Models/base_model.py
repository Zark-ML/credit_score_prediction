from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("Data/cleaned_data.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]
x_train, x_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=63)

linreg_model = LinearRegression()
linreg_model.fit(x_train, y_train)

print(linreg_model.score(x_test, y_test))
