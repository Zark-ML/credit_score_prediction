from sklearn.ensemble import IsolationForest
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

class CheckAndRemoveOutliers(DataPreprocessing):
    def __init__(self):
        pass

    def transform(self, df):
        """
        Removes outliers from a pandas DataFrame using the Isolation Forest algorithm.
        
        Parameters:
        - df: pandas DataFrame, the DataFrame from which to remove outliers.
        
        Returns:
        - clean_df: pandas DataFrame, the DataFrame after removing outliers.
        """
        # Initialize the Isolation Forest model
        isolation_forest = IsolationForest()
        
        # Fit the model on the DataFrame. Assumes the DataFrame is purely numerical.
        isolation_forest.fit(df)
        
        # Predict the anomalies (-1 for outliers, 1 for inliers)
        anomalies = isolation_forest.predict(df)

        outlier_count = (anomalies == -1).sum()
    
        logger.info(f"Number of outliers in dataframe: {outlier_count}")
        
        clean_df = df[anomalies == 1]
    
        return clean_df
