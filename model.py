from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesRegressor

class Model:
    def __init__(self):
        best_params = {'n_estimators': 50, 'criterion': 'absolute_error', 'max_features': None, 'bootstrap': False,
                       'random_state': 42}
        self.scaler = RobustScaler()
        self.model = ExtraTreesRegressor(**best_params)

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


