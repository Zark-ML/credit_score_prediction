import pickle
from helper import logger
import os
from numpy import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

import os
import pickle
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

class Model:
    """
    A base class for machine learning models.

    Parameters:
    - name (str): The name of the model.

    Attributes:
    - name (str): The name of the model.
    - _is_trained (bool): Indicates whether the model has been trained or not.
    - model: The machine learning model.
    - y_pred: The predicted values.

    Methods:
    - train(X_train, y_train): Trains the model.
    - predict(X_test): Makes predictions using the trained model.
    - score(y_test, score_type="R2"): Computes the evaluation score of the model.
    - cross_validate(X, y, cv=5, scoring='r2'): Performs cross-validation on the model.
    - save(path=None): Saves the model to a file.
    - load(path=None): Loads the model from a file.
    """

    def __init__(self, name: str):
        """
        Initializes a new instance of the Model class.

        Parameters:
        - name (str): The name of the model.
        """
        self.name = name
        self._is_trained = False
        self.model = None
        self.y_pred = None

    @property
    def trained(self):
        """
        Checks if the model has been trained.

        Returns:
        - bool: True if the model has been trained, False otherwise.
        """
        if self._is_trained:
            return True
        else:
            logger.error(f'{self} is not trained yet')
            return False

    def train(self, X_train, y_train):
        """
        Trains the model using the given training data.

        Parameters:
        - X_train: The input features for training.
        - y_train: The target values for training.
        """
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info(f"Model was trained successfully: status:{self._is_trained}")

    def predict(self, X_test):
        """
        Makes predictions using the trained model.

        Parameters:
        - X_test: The input features for prediction.

        Returns:
        - The predicted values.
        """
        if not self._is_trained:
            logger.error("Model has not been trained yet")
            return None 
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def score(self, y_test ,score_type="R2"):
        """
        Computes the evaluation score of the model.

        Parameters:
        - y_test: The true target values.
        - score_type (str): The type of score to compute. Default is "R2".

        Returns:
        - The evaluation score.
        """
        if self.y_pred is None:
            logger.error("Model has not made predictions yet")
            return None

        """
        score_types: MAE, MSE, RMSE, R2, MAPE 
        """
        match score_type:
            case "MAE":
                mae = mean_absolute_error(y_test, self.y_pred)
                return mae
            case "MSE":
                mse = mean_squared_error(y_test, self.y_pred)
                return mse
            case "RMSE":
                rmse = sqrt(mean_squared_error(y_test, self.y_pred))
                return rmse
            case "R2":
                r2 = r2_score(y_test, self.y_pred)
                return r2
            case "MAPE":
                mape = mean_absolute_percentage_error(y_test, self.y_pred)
                return mape

    def cross_validate(self, X, y, cv=5, scoring='r2'):
        """
        Performs cross-validation on the model.

        Parameters:
        - X: The input features.
        - y: The target values.
        - cv (int): The number of cross-validation folds. Default is 5.
        - scoring (str): The scoring metric to use. Default is 'r2'.

        Returns:
        - The mean cross-validation score.
        """
        if self.model is None:
            logger.error(f"{self} is not trained yet")
            return None
        return cross_val_score(self.model, X, y, cv=cv, scoring=scoring).mean()

    def save(self, path=None):
        """
        Saves the model to a file.

        Parameters:
        - path (str): The path to save the model. If not provided, a default path will be used.
        """
        if path is None:
            os.makedirs("saved_models", exist_ok=True)
            path = f"saved_models/{self.name}.pkl"

        try:
            with open(path, 'wb') as file:
                pickle.dump(self.model, file)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")

    def load(self, path=None):
        """
        Loads the model from a file.

        Parameters:
        - path (str): The path to load the model from. If not provided, a default path will be used.

        Returns:
        - The loaded model.
        """
        if path is None:
            path = f"saved_models/{self.name}.pkl"
        try:
            with open(path, 'rb') as file:
                self.model = pickle.load(file)
            logger.info(f"Model loaded from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load the model: {e}")
            return None

    def __str__(self):
        """
        Returns a string representation of the model.

        Returns:
        - The name of the model.
        """
        return self.name

