from pipeline import Pipeline
from Models.RidgeRegression import RidgeRegression
from Models.LinearRegression import LinearRegressionModel
import pandas as pd

data = pd.read_csv("Data/cleaned_data.csv")
# pipeline = Pipeline(data, RidgeRegression("RidgeRegression", alpha = 0.5, max_iter = 100))
pipeline = Pipeline(data, LinearRegressionModel())
pipeline.fit_transform()
new_data = pd.read_csv("Data/test_100.csv")
pipeline.predict(new_data)
