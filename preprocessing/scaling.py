import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from helper import logger
from preprocessing.abstract_prep import DataPreprocessing


logger.info("Successfully imported file")

class Scaling(DataPreprocessing):
    """
    This class represents the scaling preprocessing step for data.
    It inherits from the DataPreprocessing abstract class.
    """

    def __init__(self, name: str):
        """
        Initializes an instance of the Scaling class.

        Args:
            name (str): The name of the scaling instance.
        """
        super().__init__(name)

    def transform(self, target_column):
        """
        Applies scaling to the data.

        Args:
            target_column: The target column to be excluded from scaling.
        """
        logger.info(f"{self} is starting scaling")  
        
        print("Scaling with StandartScaler")      
        scaler = StandardScaler()
        scaler.fit(self.data.drop(target_column, axis=1))

        with open("logger/scaler_values.pkl", "wb") as file:
            pickle.dump(f"expectation {scaler.mean_ }, deviation {scaler.scale_}", file)
        
        scaled_features = scaler.transform(self.data.drop(target_column, axis=1))
        updated = pd.DataFrame(scaled_features, columns=self.data.columns[:-1])
        updated[target_column] = self.data[target_column]

        logger.info(f"{self} scaling ended") 

        
    


                
            
        