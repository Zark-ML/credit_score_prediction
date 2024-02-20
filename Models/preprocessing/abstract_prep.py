# Model Base Class
from abc import ABC, abstractmethod
from helper import logger

class DataPreprocessing(ABC):
    def __init__(self, name: str):
        self.name = name
        self.__is_trained = False
        self.model = None

    def mark_as_trained(self):
        self.is_trained = True
        logger.info(f'{self.name} is marked as trained')
        
    @abstractmethod
    def apply(self):
        pass

    # @abstractmethod
    # def apply_update(self):
    #     pass
    