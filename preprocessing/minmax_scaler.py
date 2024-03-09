import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'MinMaxScaler' file")

class MinMaxScaling(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.name = "MinMaxScaler"
        self.data = data
          
    def transform(self):
        logger.info(f"{self} is starting")  
        scaler = MinMaxScaler()
        columns = self.data.columns
        scaled = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled, columns=columns)
        logger.info(f"{self} ended") 
        return self.data
        

        
        
    


                
            
        