import pandas as pd
from sklearn.preprocessing import StandardScaler
from helper import logger
from Models.preprocessing.abstract_prep import DataPreprocessing

class Scaling(DataPreprocessing):
    def __init__(self, name: str, data:pd.DataFrame):
        super().__init__(name)
        self.data = data
          
    def transform(self,target_column):
        logger.info(f"{self} is starting scaling")  
        
        scaler = StandardScaler()
        features = self.data.drop(target_column, axis=1)
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        updated_columns = features.columns
        updated_data = pd.DataFrame(scaled_features, columns=updated_columns)
        updated_data[target_column] = self.data[target_column].values  
        
        logger.info(f"{self.name}  ended")
        self.mark_as_trained()
        
        return updated_data
        

        
        
    


                
            
        