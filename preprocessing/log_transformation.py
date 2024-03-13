import numpy as np
import pandas as pd
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'LogTransformation' file")

class LogTransformation(DataPreprocessing):
    def __init__(self):
        super().__init__("LogTransformation")
        
    def transform(self, data: pd.DataFrame ,column: pd.Series):
        logger.info(f"{self} is starting on {column}")

        transformed_data = data.copy()
        if (data[column] <= 0).any():
            logger.warning(f"Log transformation is not applied to {column} as it contains non-positive values.")
            return transformed_data
        else:
            transformed_data[column] = np.log(transformed_data[column])
            logger.info(f"{self.name} applied to {column} ended")
            return transformed_data
           
        
        
    


                
            
        