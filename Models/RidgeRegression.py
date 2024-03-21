from sklearn.linear_model import Ridge
from Models.abstract_model import Model
from helper import logger

class RidgeRegression(Model):

    class RidgeRegression:
        """
        Ridge Regression model implementation.
        """

        def __init__(self, name: str = "RidgeRegression",
                     alpha: float = 1.0,
                     fit_intercept: bool = True,
                     copy_X: bool = True,
                     max_iter: int = None,
                     tol: float = 0.001,
                     solver: str = 'auto',
                     random_state: int = 63):
            """
            Initialize a RidgeRegression object.

            Args:
            - name (str): Name of the model (default: "RidgeRegression").
            - alpha (float): Regularization strength (default: 1.0).
            - fit_intercept (bool): Whether to calculate the intercept for this model (default: True).
            - copy_X (bool): Whether to copy X before fitting (default: True).
            - max_iter (int): Maximum number of iterations for the solver (default: None).
            - tol (float): Tolerance for stopping criteria (default: 0.001).
            - solver (str): Solver to use for the optimization problem (default: 'auto').
            - random_state (int): Seed used by the random number generator (default: 63).
            """
            super().__init__(name)
            self.model = Ridge(alpha=alpha,
                               fit_intercept=fit_intercept,
                               copy_X=copy_X,
                               max_iter=max_iter,
                               tol=tol,
                               solver=solver,
                               random_state=random_state)