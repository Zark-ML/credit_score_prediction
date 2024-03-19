import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing
import json

logger.info("Successfully imported 'MinMaxScaler' file")

class MinMaxScaling(DataPreprocessing):
    def __init__(self):
        super().__init__("MinMaxScaler")
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.scaling_values = {}


    def fit(self, data: pd.DataFrame):
        """Fit the scaler to the data."""
        self.scaler.fit(data)
        self.is_fitted = True
        logger.info(f"{self} fitting ended")

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                return super().default(obj)

        for column in data.columns:
            min_value = data[column].min()
            max_value = data[column].max()
            self.scaling_values[column] = [min_value, max_value]
        # Save min and max values to a JSON file
        with open('Data/scaling_values.json', 'w') as json_file:
            json.dump(self.scaling_values, json_file, indent=4, cls=CustomEncoder)
        
        return data

    def transform(self, data: pd.DataFrame):
        if not self.is_fitted:
            logger.error(f"{self} is not fitted yet. Please call 'fit' with training data before transforming.")

        # Set scaler's data_min_ and data_max_ attributes
        mins = []
        maxes = []
        for value in self.scaling_values.values():
            mins.append(value[0])
            maxes.append(value[1])
        self.scaler.data_min_ = mins
        self.scaler.data_max_ = maxes
        self.scaler.fit(data)
        logger.info(f"{self} is starting transformation")
        scaled_data = self.scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        logger.info(f"{self} transformation ended")
        
        return scaled_data

    def inverse_transform(self, scaled_data: np.ndarray):
        """Inverse transform the scaled data to its original scale."""
        with open("Data/scaling_values.json", "r") as f:
            self.scaling_values = json.load(f)
        min = self.scaling_values["CREDIT_SCORE"][0]
        max = self.scaling_values["CREDIT_SCORE"][1]
        descaled_data = scaled_data * (max - min) + min

        return list(descaled_data)
