from sklearn.model_selection import train_test_split
import pandas as pd
from helper import logger
from preprocessing.check_nans import CheckNans 
from preprocessing.minmax_scaler import MinMaxScaling
from preprocessing.check_and_remove_outliers import CheckAndRemoveOutliers
import json


class Pipeline:
    def __init__(self, data: pd.DataFrame, model):
        self.data = data
        self.model = model
        self.nan_checker = CheckNans()
        self.scaler = MinMaxScaling()
        self.outlier_remover = CheckAndRemoveOutliers()

    def data_preprocessing(self, data_to_process):
        logger.info("Data Preprocessing")
        processed_data = self.nan_checker.transform(data_to_process)
        # print(processed_data.shape, "after nan checking")
        # if not hasattr(self.scaler, "min_"):
        #     self.scaler.fit(processed_data)
        processed_data = self.outlier_remover.transform(processed_data)
        # print(processed_data.shape, "after outliers removal")
        processed_data = self.scaler.transform(processed_data)
        # print(processed_data.shape, "after scaling")
        logger.info("Data Preprocessing completed")
        return pd.DataFrame(processed_data, columns=data_to_process.columns)
        
    def fit_transform(self):
        logger.info("Training the model")
        y = self.data["CREDIT_SCORE"]
        columns = pd.read_json("Data/selected_features_15.json")[0]
        X = self.data[columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train["CREDIT_SCORE"] = y_train
        df = MinMaxScaling().fit(X_train)
        df_preprocessed = self.data_preprocessing(df)
        # print(df_preprocessed, "preprocessed df of train", df_preprocessed.shape)
        X_train = df_preprocessed.drop("CREDIT_SCORE", axis=1)
        y_train = df_preprocessed["CREDIT_SCORE"]
        X_test["CREDIT_SCORE"] = y_test
        X_test_preprocessed = self.data_preprocessing(X_test)
        # print(X_test_preprocessed, X_test_preprocessed.shape, "test set")
        X_test = X_test_preprocessed.drop("CREDIT_SCORE", axis=1)
        y_test = X_test_preprocessed["CREDIT_SCORE"]
        self.model.train(X_train, y_train)
        self.model.predict(X_test)
        scores = self.get_scores(y_test)
        logger.info(f"Scores: {scores}")
        logger.info("Training completed")
        return scores
        
    def predict(self, new_data):
        logger.info("Predicting new data")
        if not self.model.trained:
            logger.error("Model is not trained. Please train the model before prediction.")
            return None
        columns = pd.read_json("Data/selected_features_15.json")[0]
        X = new_data[columns]
        X["CREDIT_SCORE"] = new_data["CREDIT_SCORE"]
        processed_data = self.data_preprocessing(X)
        X = processed_data.drop(columns=["CREDIT_SCORE"], axis=1)
        y = processed_data["CREDIT_SCORE"]
        prediction = self.model.predict(X)
        with open("Data/descaled_predictions.json", "w") as file:
            json.dump(prediction, file)
        y_descaled = self.scaler.inverse_transform(y)
        scores = self.get_scores(y_descaled)
        logger.info(f"Scores: {scores}")
        return scores
    
    def get_scores(self, y_test):
        scores = {score_type: self.model.score(y_test, score_type) for score_type in ["MAE", "MSE", "RMSE", "R2", "MAPE"]}
        return scores

    def save(self, path=None):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)

    def __str__(self):
        return str(self.model)
