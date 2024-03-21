from sklearn.ensemble import RandomForestRegressor
from helper import logger
from Models.abstract_model import Model

class RandomForestReg(Model):

    class RandomForestRegression:
        """
        A class representing a Random Forest Regression model.
        """

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
            """
            Initialize the RandomForestRegression object.

            Parameters:
            -----------
            name : str, optional
                The name of the model. Default is "RandomForestRegression".
            n_estimators : int, optional
                The number of trees in the forest. Default is 100.
            criterion : str, optional
                The function to measure the quality of a split. Default is 'friedman_mse'.
            max_depth : int, optional
                The maximum depth of the tree. Default is None.
            min_samples_split : int, optional
                The minimum number of samples required to split an internal node. Default is 2.
            min_samples_leaf : int, optional
                The minimum number of samples required to be at a leaf node. Default is 1.
            min_weight_fraction_leaf : float, optional
                The minimum weighted fraction of the sum total of weights required to be at a leaf node. Default is 0.0.
            max_features : str, optional
                The number of features to consider when looking for the best split. Default is 'sqrt'.
            max_leaf_nodes : int, optional
                Grow trees with max_leaf_nodes in best-first fashion. Default is None.
            min_impurity_decrease : float, optional
                A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
            bootstrap : bool, optional
                Whether bootstrap samples are used when building trees. Default is True.
            oob_score : bool, optional
                Whether to use out-of-bag samples to estimate the R^2 on unseen data. Default is False.
            n_jobs : int, optional
                The number of jobs to run in parallel for both fit and predict. Default is None.
            random_state : int, optional
                Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node. Default is 63.
            verbose : int, optional
                Controls the verbosity when fitting and predicting. Default is 0.
            ccp_alpha : float, optional
                Complexity parameter used for Minimal Cost-Complexity Pruning. Default is 0.0.
            max_samples : int, optional
                If bootstrap is True, the number of samples to draw from X to train each base estimator. Default is None.
            """
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

