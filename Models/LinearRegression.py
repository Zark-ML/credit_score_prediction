from sklearn.linear_model import LinearRegression
from Models.abstract_model import Model
from helper import logger


class LinearRegressionModel(Model):
    """
    A class representing a linear regression model.
    """

    def __init__(self, name: str = "LinearRegression"):
        """
        Initializes the LinearRegressionModel object.

        Args:
            name (str): The name of the model.
        """
        super().__init__(name)
        self.model = LinearRegression()
