import pandas as pd
from sklearn.ensemble import IsolationForest
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

class CheckOutliers(DataPreprocessing):
    def __init__(self, df):
        self.df = df

    def transform(self):
        """
        Removes outliers from a pandas DataFrame using the Isolation Forest algorithm.
        
        Parameters:
        - df: pandas DataFrame, the DataFrame from which to remove outliers.
        - contamination_factor: float, the proportion of outliers in the data set. Defaults to 0.01.
        - random_state: int, controls the randomness of the estimator. Defaults to 42 for reproducibility.
        
        Returns:
        - clean_df: pandas DataFrame, the DataFrame after removing outliers.
        """
        # Initialize the Isolation Forest model
        isolation_forest = IsolationForest()
        
        # Fit the model on the DataFrame. Assumes the DataFrame is purely numerical.
        isolation_forest.fit(self.df)
        
        # Predict the anomalies (-1 for outliers, 1 for inliers)
        anomalies = isolation_forest.predict(self.df)

        outlier_count = (anomalies == -1).sum()
    
        logger.info(f"Number of outliers in dataframe {outlier_count}")
        clean_df = self.df[anomalies == 1]
    
        return clean_df


            