from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


def compute_rolling_std(X_df, features, time_window, center=False):
    for feature in features:
        name = "_".join([feature, time_window, "std"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).std()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def add_features(X_df):
    Alfven_Mach_number = X_df['V'] * 1e12 * np.sqrt(X_df['Np'] * 1.7e-27 * 1e6) * np.sqrt(4e-7*np.pi) / X_df['B']
    raw_pressure = X_df['V']**2 * X_df['Np'] * 1.7e-27 * 1e12 * 1e9
    X_df['AMach_number'] = Alfven_Mach_number
    X_df['raw_pressure'] = raw_pressure
    return X_df

class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = add_features(X)
        return compute_rolling_std(X, ["Beta", "Vth", "B", "Bx", "Bz"], "2h")

def get_estimator():

    feature_extractor = FeatureExtractor()
    classifier = LogisticRegression(max_iter=1000)
    pipe = make_pipeline(feature_extractor, StandardScaler(), classifier)
    return pipe