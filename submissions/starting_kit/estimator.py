from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def compute_rolling_std(X_df, features, time_window, center=False):
    for feature in features:
        name = "_".join([feature, time_window, "std"])
        X_df[name] = X_df[feature].rolling(time_window, center=center).std()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df

def compute_rolling_mean(X_df, features, time_window, center=False):
    for feature in features:
        name = '_'.join([feature, time_window, 'mean'])
        X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
        X_df[name] = X_df[name].ffill().bfill()
        X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def add_features(X_df):

    Alfven_Mach_number = X_df['V'] * 1e12 * np.sqrt(X_df['Np_nl'] * 1.7e-27 * 1e6) * np.sqrt(4e-7*np.pi) / X_df['B']
    X_df['AMach_number'] = Alfven_Mach_number
    X_df['log_Pdyn'] = np.log(X_df.Pdyn)

    """delete columns with very low variances"""
    X_df = X_df.drop(['Range F 14'], axis=1)
    X_df = X_df.drop(['Pdyn'], axis=1)

    raw_pressure = X_df['V']**2 * X_df['Np_nl'] * 1.7e-27 * 1e12 * 1e9
    X_df['raw_pressure'] = raw_pressure
    mu_0, m_p = 1.26e-6, 1.67e-27
    Alfven_speed = X_df['B'] / (np.sqrt(mu_0 * m_p * X_df['Np_nl']))
    Alfven_Mach_number_2 = Alfven_speed / X_df['V']
    fast_mach_number = X_df['V'] / np.sqrt(340**2 + Alfven_speed**2)
    X_df['AMach_number_2'] = Alfven_Mach_number_2
    X_df['Alfven_speed'] = Alfven_speed
    X_df['fast_mach_number'] = fast_mach_number
    X_df['ratio_density_proton_alpha'] = X_df['Np_nl'] / X_df['Na_nl']
    return X_df

class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = add_features(X)

        columns = ['B','Beta', 'Vth', 'Bx', 'By', 'Bz', 'RmsBob', 'Vx', 'V', 'Vth',
                   'AMach_number', 'AMach_number_2', 'Alfven_speed']
        for t in ['2h','5h','10h','15h','20h'] : 
            X = compute_rolling_std(X, columns, t)
            X = compute_rolling_mean(X, columns, t)
        for col in columns: 
            X[col+'_mean_delta_6h'] = X[col+'_2h_mean'] - X[col+'_2h_mean'].shift(36)
        return X
    
    
class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        
        self.clf = HistGradientBoostingClassifier(max_iter=1000, max_leaf_nodes=15, l2_regularization=5e4, min_samples_leaf=150)
    def fit(self, X, y):
        return self.clf.fit(X, y)
    
    def predict(self, X):
        proba = self.clf.predict_proba(X)
        y_pred = pd.Series(proba[:, 1])
        y_pred_smoothed = y_pred.rolling(12, min_periods=0, center=True).quantile(0.90)
        proba_smoothed =  np.swapaxes(np.array([1 - y_pred_smoothed, y_pred_smoothed]), 1, 0)
        return np.argmax(proba_smoothed, axis=0)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def get_estimator():

    feature_extractor = FeatureExtractor()
    classifier = MyClassifier()
    pipe = make_pipeline(feature_extractor, StandardScaler(), classifier)
    
    return pipe

