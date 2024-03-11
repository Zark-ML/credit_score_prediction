from Models.abstract_model import Model
from sklearn.model_selection import train_test_split
import pandas as pd
from helper import logger
from preprocessing.check_nans import CheckNans 
from preprocessing.minmax_scaler import MinMaxScaler 
from preprocessing.remove_outliers import RemoveOutliers 
from Models.RidgeRegression import RidgeRegressionModel
class Pipeline:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = RidgeRegressionModel("RidgeRegression_pipeline", alpha = 0.5, max_iter = 100)

    def data_preprocessing(self):
        logger.info("Data Preprocessing")
        self.data = CheckNans(self.data)
        self.data = MinMaxScaler(self.data)
        
    def fit_transform(self, data: pd.DataFrame):
        self.data = self.data_preprocessing(data)
        y = self.data["CREDIT_SCORE"]
        columns = pd.read_json("Data/selected_features_5.json")[0]
        X = self.data[columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.train(X_train, y_train)
        self.model.predict(X_test)
        scores = list()
        scores_list = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
        for score in scores_list:
            scores.append(self.model.score(y_test, score))
        logger.info(f"Scores: {scores_list}:{scores}")
        

    def dereferance(self):
        pass
    
        # processed_data = data
        # for step in self.steps:
        #     processed_data = step.transform(processed_data)
        # 
        # self.model.train(X_train, y_train)
        # self.test_data = (X_test, y_test)

    def save(self, path=None):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)

    def predict(self):
        if not self.model.trained:
            logger.error("Model must be trained before prediction.")
        

        
        processed_data = data
        for step in self.steps:
            processed_data = step.transform(processed_data)
        
        return self.model.predict(processed_data)

    def add_step(self, step):
        self.steps.append(step)
