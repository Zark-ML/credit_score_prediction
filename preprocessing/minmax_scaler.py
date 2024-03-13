import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'MinMaxScaler' file")
class MinMaxScaler(DataPreprocessing):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False 

    def fit(self, data):
        """Fit the scaler to the data."""
        logger.info(f"{self.__class__.__name__} is fitting")
        self.scaler.fit(data)
        self.is_fitted = True
        logger.info(f"{self.__class__.__name__} fitting ended")

    def transform(self, data):
        """Transform the data using the already fitted scaler."""
        if not self.is_fitted:
            logger.error(f"{self.__class__.__name__} is not fitted yet. Please call 'fit' with training data before transforming.")
            return None

        logger.info(f"{self.__class__.__name__} is starting transformation")
        scaled_data = self.scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        logger.info(f"{self.__class__.__name__} transformation ended")
        return scaled_data
