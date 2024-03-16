import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported file")

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

