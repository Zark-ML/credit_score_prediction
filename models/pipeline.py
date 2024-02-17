from abstract_model import Model
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, steps, model: Model):
        self.steps = steps
        self.model = model

    def fit_transform(self, data, label):

        processed_data = data
        for step in self.steps:
            processed_data = step.transform(processed_data)
        X_train, X_test, y_train, y_test = train_test_split(processed_data, label, test_size=0.2)
        self.model.train(X_train, y_train)
        self.test_data = (X_test, y_test)

    def save(self, path=None):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)

    def predict(self, data):
        if not self.model.__is_train():
            raise Exception("Model must be trained before prediction.")
        
        processed_data = data
        for step in self.steps:
            processed_data = step.transform(processed_data)
        
        return self.model.predict(processed_data)

    def add_step(self, step):
        self.steps.append(step)
