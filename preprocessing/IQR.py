from helper import logger
import pandas as pd 
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'RemoveOutliers' file")

class RemoveOutliers(DataPreprocessing):
    def __init__(self):
        super().__init__("RemoveOutliers")

    def transform(self, data: pd.DataFrame):
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