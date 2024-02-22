# Model Base Class
from abc import ABC, abstractmethod
from helper import logger

class DataPreprocessing(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def transform(self, data):
        pass

    # @abstractmethod
    # def apply_update(self):
    #     pass
    