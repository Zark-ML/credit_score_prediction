import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from helper import logger

class DataPreprocessing:
    def __init__(self, data:pd.DataFrame):
        self.data = data
        
    def checkNans(self):
        logger.info(f"{self} is checking nans in dataframe")
        
        if(sum(self.data.isna().sum()) == 0):
            print("No nans in dataframe")
        else:
            print("Nans are found in dataframe")
            print("Please choose variant replace nans with mean of columns or drop them (drop or mean,median,most_frequent,constant) : ")
    
            strategies = np.array["mean",'median',"most_frequent",'constant']
            variant = input().lower()
            if variant in strategies:
                imputer = SimpleImputer(strategy=variant)
                return pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns, index=self.data.index)
            else:
                print("Invalid input. Returning original data.")
                return self.data
            
        logger.info(f"{self} is checking nans in dataframe ended")
        
    def scaling(self,target_column):
        logger.info(f"{self} is starting scaling")  
        
        print("Scaling with StandartScaler")      
        scaler = MinMaxScaler()
        scaler.fit(self.data.drop(target_column,axis=1))
        scaled_features = scaler.transform(self.data.drop(target_column, axis=1))
        updated = pd.DataFrame(scaled_features,columns=self.data.columns[:-1])
        updated[target_column] = self.data[target_column]
        print(updated.head())
        
        logger.info(f"{self} scaling ended") 
        
    def removeOutliers(self):
        logger.info(f"{self} is removing outliers from dataframe")

        cleaned_data = self.data.copy()
        for column in self.data.columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)]

        logger.info(f"{self} removed outliers from dataframe")
        
        return cleaned_data
    
    def applyLogTransformation(self, column):
        logger.info(f"{self} is applying log transformation to {column}")

        if (self.data[column] <= 0).any():
            logger.warning(f"Log transformation is not applied to {column} as it contains non-positive values.")
            return self.data
        else:
            self.data[column] = np.log(self.data[column])
            logger.info(f"Log transformation applied to {column}")
            return self.data


                
            
        