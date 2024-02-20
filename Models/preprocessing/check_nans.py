import pandas as pd
from Models.preprocessing.abstract_prep import DataPreprocessing
from helper import logger
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
class CheckNans(DataPreprocessing):
    def __init__(self, name: str, data: pd.DataFrame):
        super().__init__(name)
        self.data = data

    def apply(self):
        logger.info(f"{self} is checking nans in dataframe")

        if self.data.isna().sum().sum() == 0:
            logger.info("No nans in dataframe")
            return self.data
        else:
            logger.info("Nans are found in dataframe")
            imp_mean = IterativeImputer()
            
            cleaned_data = pd.DataFrame(imp_mean.fit_transform(self.data), columns=self.data.columns, index=self.data.index)
                    
            logger.info(f"Nans in dataframe handled using IterativeImputer strategy")
            return cleaned_data
            
