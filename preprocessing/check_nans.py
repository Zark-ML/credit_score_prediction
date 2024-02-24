import pandas as pd
from helper import logger
from sklearn.impute import SimpleImputer
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'CheckNans' file")


class CheckNans(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.data = data

    def transform(self):
            logger.info(f"{self} is checking nans in dataframe")
            
            if(sum(self.data.isna().sum()) == 0):
                logger.info("No nans in dataframe")
            else:
                logger.info("Nans are found in dataframe")
                return pd.DataFrame(SimpleImputer("mean").fit_transform(self.data), columns=self.data.columns, index=self.data.index)
                
                
            logger.info(f"{self} is checking nans in dataframe ended")