import pandas as pd
from helper import logger
from Models.preprocessing.abstract_prep import DataPreprocessing

class RemoveOutliers(DataPreprocessing):
    def __init__(self, name: str, data: pd.DataFrame):
        super().__init__(name)
        self.data = data

    def apply(self, iqr_multiplier=1.5):
        logger.info(f"{self} is removing outliers from dataframe")
        
        outlier_indices = []

        for column in self.data.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)

            column_outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)].index
            outlier_indices.extend(column_outliers)

        outlier_indices = list(set(outlier_indices))
        cleaned_data = self.data.drop(outlier_indices)

        logger.info(f"Removed outliers from dataframe. Original rows: {self.data.shape[0]}, After cleaning: {cleaned_data.shape[0]}")
        self.mark_as_trained()

        return cleaned_data
    
        
        
