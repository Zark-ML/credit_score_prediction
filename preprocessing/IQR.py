import pandas as pd 
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'RemoveOutliers' file")

class RemoveOutliers(DataPreprocessing):
    """
    A class that removes outliers from a given dataset using the Interquartile Range (IQR) method.
    """

    def __init__(self):
        """
        Initializes an instance of the RemoveOutliers class.
        
        Parameters:
            None
        """
        super().__init__("RemoveOutliers")

    def transform(self, data: pd.DataFrame):
        """
        Applies the outlier removal transformation on the given dataset.
        
        Parameters:
            data (pd.DataFrame): The input dataset to be transformed.
        
        Returns:
            pd.DataFrame: The transformed dataset with outliers removed.
        """
        logger.info(f"{self} is starting")

        cleaned_data = data.copy()
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]

        logger.info(f"{self} ended")
        
        return cleaned_data
