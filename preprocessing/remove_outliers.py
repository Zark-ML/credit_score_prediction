from helper import logger
import pandas as pd 
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'RemoveOutliers' file")

class RemoveOutliers(DataPreprocessing):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def transform(self):
        logger.info(f"{self} is removing outliers from dataframe")

        cleaned_data = self.data.copy()
        for column in self.data.columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]

        logger.info(f"{self} removed outliers from dataframe")
        
        return cleaned_data