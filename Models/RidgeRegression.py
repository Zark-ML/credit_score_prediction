from sklearn.linear_model import Ridge
from Models.abstract_model import Model
from helper import logger

class RidgeRegression(Model):

    def __init__(self, name: str = "RidgeRegression",
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 copy_X: bool = True,
                 max_iter: int = None,
                 tol: float = 0.001,
                 solver: str = 'auto',
                 random_state: int = 63):
        super().__init__(name)
        self.model = Ridge(alpha=alpha,
                           fit_intercept=fit_intercept,
                           copy_X=copy_X,
                           max_iter=max_iter,
                           tol=tol,
                           solver=solver,
                           random_state=random_state)

 