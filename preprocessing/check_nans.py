import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing

logger.info("Successfully imported file")


class CheckNans(DataPreprocessing):
    def __init__(self,name:str):
        super().__init__(name)


    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        logger.info(f"{self} is checking nans in dataframe")

        if data.isna().sum().sum() == 0:
            logger.info("No nans in dataframe")
            return data
        else:
            logger.info("Nans are found in dataframe")
            imp_mean = IterativeImputer()
            
            cleaned_data = pd.DataFrame(imp_mean.fit_transform(data), columns=data.columns, index=data.index)
                    
            logger.info(f"Nans in dataframe handled using IterativeImputer strategy")
            return cleaned_data

    def __str__(self) -> str:
        return super().__str__()