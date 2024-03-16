import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing
import json
from sklearn.model_selection import train_test_split

logger.info("Successfully imported 'MinMaxScaler' file")
class MinMaxScaling(DataPreprocessing):
    def __init__(self):
        super().__init__("MinMaxScaler")
        self.scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame, catboost):
        """Fit the scaler to the data."""
        logger.info(f"{self} is fitting")
        if catboost:
            self.scaler.fit(data)
            min_scores = data.min(axis=0)
            max_scores = data.max(axis=0)
            json_dict = dict()
            for i in range(data.shape[1]):
                json_dict[data.columns[i]] = [min_scores[i], max_scores[i]]

            with open("Data/scaling_values.json", "w") as json_file:
                json.dump(json_dict, json_file, indent=4)
        self.scaler.fit(data)
        self.is_fitted = True
        logger.info(f"{self} fitting ended")


    def transform(self, data:pd.DataFrame, catboost=False):
        # """Transform the data using the already fitted scaler."""
        # with open("Data/scaling_values.json", "r") as f:
        #     scaling_values = json.load(f)

        # # Extract min and max values for each column
        # columns_min_max = {col: scaling_values[col] for col in data.columns}

        # # Set scaler's data_min_ and data_max_ attributes
        # self.scaler.data_min_ = [columns_min_max[col][0] for col in data.columns]
        # self.scaler.data_max_ = [columns_min_max[col][1] for col in data.columns]
        self.fit(data, catboost=catboost)
        
        if not self.is_fitted:
            logger.error(f"{self} is not fitted yet. Please call 'fit' with training data before transforming.")
            return None

        logger.info(f"{self} is starting transformation")
        scaled_data = self.scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        logger.info(f"{self} transformation ended")
        # print(scaled_data)
        return scaled_data
    
    def inverse_transform(self, scaled_data: np.ndarray):
        """Inverse transform the scaled data to its original scale."""
        with open("Data/scaling_values.json", "r") as f:
            scaling_values = json.load(f)
        min, max = scaling_values["CREDIT_SCORE"][0], scaling_values["CREDIT_SCORE"][1]
        descaled_data = scaled_data * (max-min) + min
        return list(descaled_data)
