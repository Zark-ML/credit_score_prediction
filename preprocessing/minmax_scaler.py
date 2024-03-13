import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'MinMaxScaler' file")
class MinMaxScaling(DataPreprocessing):
    def __init__(self):
        super().__init__("MinMaxScaler")
        self.scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame):
        """Fit the scaler to the data."""
        logger.info(f"{self} is fitting")
        self.scaler.fit(data)
        self.is_fitted = True
        logger.info(f"{self} fitting ended")

    def transform(self, data:pd.DataFrame):
        """Transform the data using the already fitted scaler."""
        self.fit(data)
        
        if not self.is_fitted:
            logger.error(f"{self} is not fitted yet. Please call 'fit' with training data before transforming.")
            return None

        logger.info(f"{self} is starting transformation")
        scaled_data = self.scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        logger.info(f"{self} transformation ended")
        return scaled_data
