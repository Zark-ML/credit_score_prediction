from sklearn.linear_model import LinearRegression
from Models.abstract_model import Model
from helper import logger


class LinearRegression(Model):
    
    def __init__(self, name: str):
        super().__init__(name)
        self.model = LinearRegression()

  