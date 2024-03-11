from itertools import product
import numpy as np
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm 

def evaluate_model(model_class, x_train, y_train, **kwargs):
    return model_class(f"{model_class} evaluation", **kwargs).cross_validate(x_train, y_train, cv=5).mean()

def grid_search(model_class, x_train, y_train, kwargs_values):
    keys, values = zip(*kwargs_values.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]


    best_score = -1 * np.inf
    best_params = dict()

    for kwargs in tqdm(combinations, desc=f"Grid search for {model_class}"):
        result = evaluate_model(model_class, x_train, y_train, **kwargs)
        if result > best_score:
            best_score = result
            best_params = kwargs
    
    best_params['best_score'] = best_score
    return best_params
