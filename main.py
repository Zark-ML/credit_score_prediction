from pipeline import Pipeline
from Models import RidgeRegression
import pandas as pd

data = pd.read_csv("Data/cleaned_data.csv")
pipeline = Pipeline(data)
pipeline.fit_transform(data)
