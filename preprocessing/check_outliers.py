import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported file")

class CheckOutliers(DataPreprocessing):
    def init(self,name:str):
        super().__init__(name)
        
    def transform(self, data:pd.DataFrame) -> None:
        logger.info("Checking outliers")
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        iso_forest.fit(data)
        outlier_predictions = iso_forest.predict(data)
        num_outliers = np.sum(outlier_predictions == -1)
        logger.info(f"Number of outliers in the dataset: {num_outliers}")


    def __str__(self) -> str:
        return super().__str__()