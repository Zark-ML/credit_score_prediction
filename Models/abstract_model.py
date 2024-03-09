from abc import ABC, abstractmethod
import pickle
from helper import logger
import os
from numpy import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self._is_trained = False
        self.model = None

    @property
    def trained(self):
        if self._is_trained:
            return True
        else:
            logger.error(f'{self} is not trained yet')
            return False


    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self._is_trained = True
        logger.info(f"Model was trained successfully: status:{self._is_trained}")


    def predict(self, X_test):
        if self._is_trained:
            self.y_pred = self.model.predict(X_test)
            return self.y_pred
        else:
            logger.error("Model have not trained yet")



    def score(self, y_test ,score_type="R2"):
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
        if self.model is None:
            logger.error(f"{self} is not trained yet")
            return None
        print(cross_val_score(self.model, X, y, cv=cv, scoring=scoring))
        return cross_val_score(self.model, X, y, cv=cv, scoring=scoring).mean()

    
    def save(self, path=None):
        if path is None:
            os.makedirs("saved_models", exist_ok=True)
            path = f"saved_models/{self.name}.pkl"

        try:
            with open(path, 'wb') as file:
                pickle.dump(self, file)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save the model: {e}")
        

    
    def load(self, path=None):
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
        return self.name

