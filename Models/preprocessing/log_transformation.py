import numpy as np
import pandas as pd
from helper import logger
from Models.preprocessing.abstract_prep import DataPreprocessing

class LogTransformation(DataPreprocessing):
    def __init__(self, name: str, data: pd.DataFrame):
        super().__init__(name)
        self.data = data
        
    def apply(self, column):
        logger.info(f"{self.name} is applying log transformation to {column}")

        transformed_data = self.data.copy()
        if (transformed_data[column] <= 0).any():
            logger.warning(f"Log transformation is not applied to {column} as it contains non-positive values.")
        else:
            transformed_column_name = f"{column}_log_transformed"
            transformed_data[transformed_column_name] = np.log(transformed_data[column])
            logger.info(f"Log transformation applied to {column}, result stored in {transformed_column_name}")

        return transformed_data
