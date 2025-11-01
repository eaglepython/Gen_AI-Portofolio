import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

class LGDModel:
    def __init__(self):
        self.model = CoxPHFitter()

    def train(self, df, duration_col, event_col):
        X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
        self.model.fit(X_train, duration_col=duration_col, event_col=event_col)
        print(self.model.summary)
        return self.model

    def predict(self, df):
        return self.model.predict_expectation(df)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = CoxPHFitter().load(path)
