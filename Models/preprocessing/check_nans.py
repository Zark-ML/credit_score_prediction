import pandas as pd
from helper import logger
from sklearn.impute import SimpleImputer
from Models.preprocessing.abstract_prep import DataPreprocessing

class CheckNans(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.data = data

    def transform(self):
            logger.info(f"{self} is checking nans in dataframe")
            
            if(sum(self.data.isna().sum()) == 0):
                logger.info("No nans in dataframe")
            else:
                logger.info("Nans are found in dataframe")
                logger.info("Please choose variant replace nans with mean of columns or drop them (drop or mean,median,most_frequent,constant) : ")
        
                strategies = ["mean",'median',"most_frequent",'constant']
                variant = input().lower()
                if variant in strategies:
                    imputer = SimpleImputer(strategy=variant)
                    return pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns, index=self.data.index)
                else:
                    print("Invalid input. Returning original data.")
                    return self.data
                
            logger.info(f"{self} is checking nans in dataframe ended")