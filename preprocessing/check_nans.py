import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported file")

class CheckNans(DataPreprocessing):
    """
    A class for checking and handling missing values in a DataFrame.

    Args:
        DataPreprocessing: The base class for data preprocessing.

    Attributes:
        None

    Methods:
        __init__: Initializes the CheckNans object.
        transform: Transforms the input DataFrame by handling missing values.

    """

    def __init__(self):
        """
        Initializes the CheckNans object.

        Args:
            None

        Returns:
            None

        """
        super().__init__("CheckNans")

    def transform(self, data: pd.DataFrame):
        """
        Transforms the input DataFrame by handling missing values.

        If there are no missing values in the DataFrame, it logs a message indicating that.
        If there are missing values, it replaces them with the mean value of each column.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        """
        logger.info(f"{self.__class__.__name__} is starting")

        if data.isna().sum().sum() == 0:
            logger.info("No NaNs in dataframe")
        else:
            logger.info("NaNs found in dataframe")
            imputer = SimpleImputer(strategy="mean")
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
            logger.info(f"{self.__class__.__name__} ended")
        return data

