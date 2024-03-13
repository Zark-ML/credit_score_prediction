import pandas as pd
from helper import logger
from sklearn.impute import SimpleImputer
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported 'CheckNans' file")

class CheckNans(DataPreprocessing):

    def __init__(self):
        super().__init__("CheckNans")

    def transform(self, data: pd.DataFrame):
        logger.info(f"{self.__class__.__name__} is starting")

        if data.isna().sum().sum() == 0:
            logger.info("No NaNs in dataframe")
        else:
            logger.info("NaNs found in dataframe")
            imputer = SimpleImputer(strategy="mean")
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
            logger.info(f"{self.__class__.__name__} ended")
        return data

