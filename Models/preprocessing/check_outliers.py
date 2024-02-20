import numpy as np
from Models.preprocessing.abstract_prep import DataPreprocessing
from sklearn.ensemble import IsolationForest
from helper import logger

class CheckOutliers(DataPreprocessing):
    def __init__(self, name: str):
        super().__init__(name)
        
    def apply(self,data):
        logger.info("Checking outliers")
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        iso_forest.fit(data)
        outlier_predictions = iso_forest.predict(data)
        num_outliers = np.sum(outlier_predictions == -1)
        logger.info(f"Number of outliers in the dataset: {num_outliers}" )
        
    
        
