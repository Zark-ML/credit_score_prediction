import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'StandardScaling' file")

class StandardScaling(DataPreprocessing):
    """
    This class represents the StandardScaling preprocessing technique.
    It inherits from the DataPreprocessing abstract class.
    """

    def __init__(self):
        """
        Initializes an instance of the StandardScaling class.
        """
        super().__init__("StandardScaling")
          
    def transform(self, data: pd.DataFrame):
        """
        Applies the StandardScaler to the input data.

        Args:
            data (pd.DataFrame): The input data to be transformed.

        Returns:
            pd.DataFrame: The transformed data.
        """
        logger.info(f"{self} is starting")  
        scaler = StandardScaler()
        columns = data.columns
        scaled = scaler.fit_transform(data)
        data = pd.DataFrame(scaled, columns=columns)
        logger.info(f"{self} ended") 
        return data
        

        
        
    


                
            
        