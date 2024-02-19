# Model Base Class
from abc import ABC, abstractmethod
from helper import logger

class DataPreprocessing(ABC):
    def __init__(self, name: str):
        self.name = name
        self.__is_trained = False
        self.model = None

    def __is_train(self):
        if self.__is_trained:
            return True
        else:
            logger.error(f'{self} is not trained yet')
            return False

    @abstractmethod
    def train(self, data, label, test_size=0.2):
        pass