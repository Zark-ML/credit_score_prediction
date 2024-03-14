import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing
import json

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
        with open("Data/scaling_values.json", "r") as f:
            scaling_values = json.load(f)

        # Extract min and max values for each column
        columns_min_max = {col: scaling_values[col] for col in data.columns}

        # Set scaler's data_min_ and data_max_ attributes
        self.scaler.data_min_ = [columns_min_max[col][0] for col in data.columns]
        self.scaler.data_max_ = [columns_min_max[col][1] for col in data.columns]
        self.fit(data)
        
        if not self.is_fitted:
            logger.error(f"{self} is not fitted yet. Please call 'fit' with training data before transforming.")
            return None

        logger.info(f"{self} is starting transformation")
        scaled_data = self.scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        logger.info(f"{self} transformation ended")
        # print(scaled_data)
        return scaled_data
    
    def inverse_transform(self, scaled_data: pd.DataFrame):
        """Inverse transform the scaled data to its original scale."""
        with open("Data/scaling_values.json", "r") as f:
            scaling_values = json.load(f)


        # Extract min and max values for each column
        columns_min_max = {col: scaling_values[col] for col in scaled_data.columns}

        # Calculate inverse scaling
        descaled_data = pd.DataFrame()
        for col in scaled_data.columns:
            min_val, max_val = columns_min_max[col]
            descaled_data[col] = scaled_data[col] * (max_val - min_val) + min_val

        return descaled_data
