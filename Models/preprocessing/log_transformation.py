import numpy as np
import pandas as pd
from helper import logger
from Models.preprocessing.abstract_prep import DataPreprocessing

class LogTransformation(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.data = data
        
    def transform(self, column):
        logger.info(f"{self} is applying log transformation to {column}")

        transformed_data = self.data.copy()
        if (self.data[column] <= 0).any():
            logger.warning(f"Log transformation is not applied to {column} as it contains non-positive values.")
            return transformed_data
        else:
            transformed_data[column] = np.log(transformed_data[column])
            logger.info(f"Log transformation applied to {column}")
            return transformed_data
           
        
        
    


                
            
        