from abc import ABC, abstractmethod
from helper import logger
import pandas as pd

logger.info("Successfully imported 'abstract_prep' file")

class DataPreprocessing(ABC):
    def __init__(self, name: str):
        self.name = name

    
    @abstractmethod
    def transform(self, data: pd.DataFrame):
        pass

    def __str__(self):
        return f"_{self.name} preprocessing class_"