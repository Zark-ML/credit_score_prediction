from sklearn.linear_model import Lasso
from Models.abstract_model import Model
from helper import logger
class LassoRegression(Model):
    """
    LassoRegression is a class that represents a Lasso Regression model.
    """

    def __init__(self, name: str = "LassoRegression",
                 alpha: float = 1.0,
                 positive: bool = False,
                 tol: float = 0.0001,
                 max_iter: int = 1000,
                 precompute: bool = False,
                 copy_X: bool = True,
                 random_state: int = 63,
                 selection: str = 'cyclic'):
        """
        Initialize the LassoRegression object.

        Parameters:
        - name (str): The name of the model. Default is "LassoRegression".
        - alpha (float): Constant that multiplies the L1 term. Default is 1.0.
        - positive (bool): When set to True, forces the coefficients to be positive. Default is False.
        - tol (float): The tolerance for the optimization solver. Default is 0.0001.
        - max_iter (int): The maximum number of iterations. Default is 1000.
        - precompute (bool): Whether to use a precomputed Gram matrix to speed up calculations. Default is False.
        - copy_X (bool): Whether to copy X before fitting. Default is True.
        - random_state (int): The seed of the pseudo random number generator. Default is 63.
        - selection (str): The type of algorithm to use for coordinate descent. Default is 'cyclic'.
        """
        super().__init__(name)
        self.model = Lasso(alpha=alpha, 
                           positive=positive, 
                           tol=tol, 
                           max_iter=max_iter, 
                           precompute=precompute, 
                           copy_X=copy_X, 
                           random_state=random_state, 
                           selection=selection)
