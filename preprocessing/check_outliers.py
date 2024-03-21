import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported file")

class CheckOutliers(DataPreprocessing):
    """
    This class is responsible for checking outliers in a dataset.
    It inherits from the DataPreprocessing class.
    """

    def __init__(self, name: str):
        """
        Initializes the CheckOutliers object.

        Args:
            name (str): The name of the CheckOutliers object.
        """
        super().__init__(name)
        
    def transform(self, data: pd.DataFrame) -> None:
        """
        Transforms the data by checking for outliers.

        Args:
            data (pd.DataFrame): The input dataset.

        Returns:
            None
        """
        logger.info("Checking outliers")
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        iso_forest.fit(data)
        outlier_predictions = iso_forest.predict(data)
        num_outliers = np.sum(outlier_predictions == -1)
        logger.info(f"Number of outliers in the dataset: {num_outliers}")
