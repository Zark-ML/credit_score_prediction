from abc import ABC, abstractmethod
import pandas as pd
from helper import logger
import pandas as pd

logger.info("Successfully imported file")

class DataPreprocessing(ABC):
    """
    Abstract base class for data preprocessing.

    Parameters:
    -----------
    name : str
        The name of the preprocessing class.

    Methods:
    --------
    transform(data: pd.DataFrame)
        Abstract method to transform the data.

    __str__()
        Returns a string representation of the preprocessing class.
    """

    def __init__(self, name:str):
        """
        Initializes the DataPreprocessing object.

        Parameters:
        -----------
        name : str
            The name of the preprocessing class.
        """
        self.name = name

    
    @abstractmethod
    def transform(self, data: pd.DataFrame):
        """
        Abstract method to transform the data.

        Parameters:
        -----------
        data : pd.DataFrame
            The input data to be transformed.
        """
        pass

    def __str__(self):
        """
        Returns a string representation of the preprocessing class.
        """
        return f"_{self.name} preprocessing class_"
