# Model Base Class
from abc import ABC, abstractmethod
from pathlib import Path
from helper import logger

class Model(ABC):
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

    @abstractmethod
    def save(self, path=None):
        pass

    @abstractmethod
    def load(self, path=None):
        pass

    def __str__(self):
        return self.name

