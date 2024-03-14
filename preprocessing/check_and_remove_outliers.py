from sklearn.ensemble import IsolationForest
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing
import pandas as pd

class CheckAndRemoveOutliers(DataPreprocessing):
    def __init__(self):
        super().__init__("CheckAndRemoveOutliers")

    def transform(self, data: pd.DataFrame):
        """
        Removes outliers from a pandas DataFrame using the Isolation Forest algorithm.
        
        Parameters:
        - df: pandas DataFrame, the DataFrame from which to remove outliers.
        
        Returns:
        - clean_df: pandas DataFrame, the DataFrame after removing outliers.
        """
        # Initialize the Isolation Forest model
        isolation_forest = IsolationForest(random_state=42)
        
        # Fit the model on the DataFrame. Assumes the DataFrame is purely numerical.
        isolation_forest.fit(data)
        
        # Predict the anomalies (-1 for outliers, 1 for inliers)
        anomalies = isolation_forest.predict(data)

        outlier_count = (anomalies == -1).sum()
    
        logger.info(f"Number of outliers in dataframe: {outlier_count}")
        
        clean_df = data[anomalies == 1]
    
        return clean_df
