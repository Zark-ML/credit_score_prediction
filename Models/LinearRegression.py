from sklearn.linear_model import LinearRegression
from abstract_model import Model
from ..helper import logger
from numpy import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

class LinearRegressionModel(Model):
    
    def __init__(self, name: str):
        super().__init__(name)
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)
        self.__is_trained = True

    def predict(self, X_test):
        if self.__is_trained:
            self.y_pred = self.model.predict(X_test)
            return self.y_pred
        else:
            logger.error("Model have not trained yet")

    def score(self, y_test ,score_type="RMSE"):
        # MAE, MSE, RMSE, R2, MAPE
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
            