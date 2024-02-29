from abc import ABC, abstractmethod
import pandas as pd
from helper import logger

logger.info("Successfully imported file")

class DataPreprocessing(ABC):
    def __init__(self, name:str):
        self.name = name
    
    @abstractmethod
    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def __str__(self) -> str:
        return self.name