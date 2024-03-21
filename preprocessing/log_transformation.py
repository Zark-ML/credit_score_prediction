import numpy as np
import pandas as pd
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported file")

class LogTransformation(DataPreprocessing):
    """
    A class that performs log transformation on a given column of a DataFrame.

    Attributes:
        name (str): The name of the LogTransformation instance.

    Methods:
        transform(data: pd.DataFrame, column: pd.Series) -> pd.DataFrame:
            Applies log transformation to the specified column of the DataFrame.

    """

    def __init__(self):
        """
        Initializes an instance of the LogTransformation class.
        
        Parameters:
            None
        """
        super().__init__("LogTransformation")
        
    def transform(self, data: pd.DataFrame, column: pd.Series) -> pd.DataFrame:
        """
        Applies log transformation to the specified column of the DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            column (pd.Series): The column to apply log transformation on.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        """
        logger.info(f"{self} is starting on {column}")

        transformed_data = data.copy()
        if (data[column] <= 0).any():
            logger.warning(f"Log transformation is not applied to {column} as it contains non-positive values.")
            return transformed_data
        else:
            transformed_data[column] = np.log(transformed_data[column])
            logger.info(f"{self.name} applied to {column} ended")
            return transformed_data



                
            
        