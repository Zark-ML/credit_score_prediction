from abc import ABC, abstractmethod
from helper import logger

logger.info("Successfully imported 'abstract_prep' file")

class DataPreprocessing(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None

    
    @abstractmethod
    def transform(self, data):
        pass