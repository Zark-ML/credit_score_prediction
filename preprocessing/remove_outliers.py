import pandas as pd 
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported file")

class RemoveOutliers(DataPreprocessing):
    def __init__(self, name:str):
        super().__init__(name)

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        logger.info(f"{self} is removing outliers from dataframe")
        cleaned_data = data.copy()
        
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]

        logger.info(f"{self} removed outliers from dataframe")
        
        return cleaned_data
    
    def __str__(self) -> str:
        return super().__str__()