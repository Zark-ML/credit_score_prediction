from abc import ABC, abstractmethod
import pickle
from helper import logger

class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def train(self, data, label):
        pass

    @abstractmethod
    def predict(self, label):
        pass

    @abstractmethod
    def save(self, path=None):
        if path is None:
            path = f"{self.name}.pkl"

        try:
            with open(path, 'wb') as file:
                pickle.dump(self, file)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")
        

    @abstractmethod
    def load(self, path=None):
        try:
            with open(path, 'rb') as file:
                model = pickle.load(file)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            return None

    def __str__(self):
        return self.name

