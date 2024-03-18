import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'StandardScaling' file")

class StandardScaling(DataPreprocessing):
    def __init__(self):
        super().__init__("StandardScaling")
          
    def transform(self, data: pd.DataFrame):
        logger.info(f"{self} is starting")  
        scaler = StandardScaler()
        columns = data.columns
        scaled = scaler.fit_transform(data)
        data = pd.DataFrame(scaled, columns=columns)
        logger.info(f"{self} ended") 
        return data
        

        
        
    


                
            
        