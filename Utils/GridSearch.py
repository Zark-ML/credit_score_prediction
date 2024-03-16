from itertools import product
import numpy as np
from tqdm import tqdm 

def evaluate_model(model_class, x_train, y_train, **kwargs):
    """
    Evaluate the performance of a model using cross-validation.

    Args:
        model_class: The class of the model to evaluate.
        x_train: The training data.
        y_train: The target labels.
        **kwargs: Additional keyword arguments to be passed to the model.

    Returns:
        The mean score of the cross-validation.
    """
    return model_class(f"{model_class} evaluation", **kwargs).cross_validate(x_train, y_train, cv=5).mean()

def grid_search(model_class, x_train, y_train, kwargs_values):
    """
    Perform a grid search to find the best combination of hyperparameters for a model.

    Args:
        model_class: The class of the model to perform the grid search on.
        x_train: The training data.
        y_train: The target labels.
        kwargs_values: A dictionary of hyperparameter values to search over.

    Returns:
        A dictionary containing the best hyperparameters and the corresponding best score.
    """
    keys, values = zip(*kwargs_values.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    best_score = -1 * np.inf
    best_params = dict()

    for kwargs in tqdm(combinations):
        result = evaluate_model(model_class, x_train, y_train, **kwargs)
        if result > best_score:
            best_score = result
            best_params = kwargs
    
    best_params['best_score'] = best_score
    return best_params
