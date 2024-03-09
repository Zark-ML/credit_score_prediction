import pandas as pd
from helper import logger
from sklearn.impute import SimpleImputer
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'CheckNans' file")


class CheckNans(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.name = "CheckNans"
        self.data = data

    def transform(self):
            logger.info(f"{self} is starting")
            
            if(sum(self.data.isna().sum()) == 0):
                logger.info("No nans in dataframe")
                return self.data
            else:
                logger.info("Nans are found in dataframe")
                logger.info(f"{self} ended")
                return pd.DataFrame(SimpleImputer("mean").fit_transform(self.data), columns=self.data.columns, index=self.data.index)
                
                
            