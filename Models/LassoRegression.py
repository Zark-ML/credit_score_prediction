from sklearn.linear_model import Lasso
from Models.abstract_model import Model
from helper import logger
class LassoRegression(Model):

    def __init__(self, name: str,
                 alpha: float = 1.0,
                 positive: bool = False,
                 tol: float = 0.0001,
                 max_iter: int = 1000,
                 precompute: bool = False,
                 copy_X: bool = True,
                 random_state: int = 63,
                 selection: str = 'cyclic'):
        super().__init__(name)
        self.model = Lasso(alpha=alpha, 
                           positive=positive, 
                           tol=tol, 
                           max_iter=max_iter, 
                           precompute=precompute, 
                           copy_X=copy_X, 
                           random_state=random_state, 
                           selection=selection)
