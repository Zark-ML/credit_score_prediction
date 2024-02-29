import numpy as np
import pandas as pd
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported file")

class LogTransformation(DataPreprocessing):
    def __init__(self,name:str):
        super().__init__(name)
        
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
           
        
    def __str__(self) -> str:
        return super().__str__()


                
            
        