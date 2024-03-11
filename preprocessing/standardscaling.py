import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'Scaling' file")

class StandardScaling(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.name = "Scaling"
        self.data = data
          
    def transform(self):
        logger.info(f"{self} is starting")  
        scaler = StandardScaler()
        columns = self.data.columns
        scaled = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled, columns=columns)
        logger.info(f"{self} ended") 
        return self.data
        

        
        
    


                
            
        