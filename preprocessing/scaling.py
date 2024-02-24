import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported 'Scaling' file")

class Scaling(DataPreprocessing):
    def __init__(self, data:pd.DataFrame):
        self.data = data
          
    def transform(self,target_column):
        logger.info(f"{self} is starting scaling")  
        
        print("Scaling with StandartScaler")      
        scaler = StandardScaler()
        scaler.fit(self.data.drop(target_column,axis=1))
        scaled_features = scaler.transform(self.data.drop(target_column, axis=1))
        updated = pd.DataFrame(scaled_features,columns=self.data.columns[:-1])
        updated[target_column] = self.data[target_column]
        print(updated.head())
        
        logger.info(f"{self} scaling ended") 
        

        
        
    


                
            
        