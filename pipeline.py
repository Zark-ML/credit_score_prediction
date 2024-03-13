from Models.abstract_model import Model
from sklearn.model_selection import train_test_split
import pandas as pd
from helper import logger
from preprocessing.check_nans import CheckNans 
from preprocessing.minmax_scaler import MinMaxScaler 
from preprocessing.check_and_remove_outliers import CheckAndRemoveOutliers
from Models.RidgeRegression import RidgeRegression
class Pipeline:
    def __init__(self, data: pd.DataFrame, model):
        self.data = data
        self.model = model
        self.nan_checker = CheckNans("CheckNans")
        self.scaler = MinMaxScaler()
        self.outlier_remover = CheckAndRemoveOutliers()

    def data_preprocessing(self, data_to_process, fit=False):
        logger.info("Data Preprocessing")
        data_to_process = self.nan_checker.transform(data_to_process)
        if fit:
            data_to_process = self.scaler.fit_transform(data_to_process)
        else:
            data_to_process = self.scaler.transform(data_to_process)
        data_to_process = self.outlier_remover.transform(data_to_process)

        logger.info("Data Preprocessing completed")
        return data_to_process
        
    def fit_transform(self):
        y = self.data["CREDIT_SCORE"]
        columns = pd.read_json("Data/selected_features_5.json")[0]
        X = self.data[columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_preprocessed = self.data_preprocessing(X_train, fit=True)
        X_test_preprocessed = self.data_preprocessing(X_test)
        print('X_train_preprocessed', X_train_preprocessed)
        print('X_test_preprocessed', X_test_preprocessed)
        self.model.train(X_train_preprocessed, y_train)
        scores = {score_type: self.model.score(X_test_preprocessed, y_test, score_type) for score_type in ["MAE", "MSE", "RMSE", "R2", "MAPE"]}
        logger.info(f"Scores: {scores}")
        
    def predict(self, new_data):
        if not self.model.trained:
            logger.error("Model is not trained. Please train the model before prediction.")
            return None
        processed_data = self.data_preprocessing(new_data)
        return self.model.predict(processed_data)

    def save(self, path=None):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)

    def __str__(self):
        return str(self.model)
