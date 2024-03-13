from sklearn.ensemble import RandomForestRegressor
from helper import logger
from Models.abstract_model import Model

class RandomForestModel(Model):

    def __init__(self, name: str = "RandomForestRegression",
                n_estimators: int = 100,
                criterion: str = 'friedman_mse',
                max_depth: int = None,
                min_samples_split: int = 2,
                min_samples_leaf: int = 1,
                min_weight_fraction_leaf: float = 0.0,
                max_features: str = 'sqrt',
                max_leaf_nodes: int = None,
                min_impurity_decrease: float = 0.0,
                bootstrap: bool = True,
                oob_score: bool = False,
                n_jobs: int = None,
                random_state: int = 63,
                verbose: int = 0,
                ccp_alpha: float = 0.0,
                max_samples: int = None):
        super().__init__(name)
        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           criterion=criterion,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                           max_features=max_features,
                                           max_leaf_nodes=max_leaf_nodes,
                                           min_impurity_decrease=min_impurity_decrease,
                                           bootstrap=bootstrap,
                                           oob_score=oob_score,
                                           n_jobs=n_jobs,
                                           random_state=random_state,
                                           verbose=verbose,
                                           ccp_alpha=ccp_alpha,
                                           max_samples=max_samples)

