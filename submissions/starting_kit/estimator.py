from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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

def add_features(X_df):
    # drop canals of proton flux
    # X_df = X_df.drop(X_df.columns[list(range(10, 25))], axis=1)
    # X_df = X_df[['B', 'Na_nl', 'Np_nl', 'V', 'Vth', 'Beta', 'Pdyn', 'RmsBob']]

    Alfven_Mach_number = X_df['V'] * 1e12 * np.sqrt(X_df['Np_nl'] * 1.7e-27 * 1e6) * np.sqrt(4e-7*np.pi) / X_df['B']
    raw_pressure = X_df['V']**2 * X_df['Np_nl'] * 1.7e-27 * 1e12 * 1e9
    X_df['AMach_number'] = Alfven_Mach_number
    X_df['raw_pressure'] = raw_pressure
    mu_0 = 1.26e-6
    m_p = 1.67e-27
    Alfven_speed = X_df['B'] / (np.sqrt(mu_0 * m_p * X_df['Np_nl']))
    Alfven_Mach_number_2 = Alfven_speed / X_df['V']
    fast_mach_number = X_df['V'] / np.sqrt(340**2 + Alfven_speed**2)
    X_df['AMach_number_2'] = Alfven_Mach_number_2
    X_df['Alfven_speed'] = Alfven_speed
    X_df['fast_mach_number'] = fast_mach_number
    X_df['ratio_density_proton_alpha'] = X_df['Np_nl'] / X_df['Na_nl']
    # Add time information
    X_df['year'] =  X_df.index.year
    X_df['month'] =  X_df.index.month
    X_df['day'] =  X_df.index.day
    X_df['hour'] =  X_df.index.hour
    X_df['minute'] =  X_df.index.minute
    return X_df

class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = add_features(X)
        return compute_rolling_std(X, ["Beta", "Vth", "B", "Bx", "By", "Bz", "Bx_rms", "By_rms", 
                                       "Bz_rms", "V", "Vx", "Vy", "Vz", "AMach_number", 
                                       "raw_pressure", "Pdyn", "RmsBob", "Na_nl", "Np", "Np_nl"], "2h")
        # return compute_rolling_std(X, ["Beta"], "2h")
        # return compute_rolling_std(X, ["Beta", "Vth", "B", "Bx", "By", "Bz", "V", "Vx", "Vy", "Vz", 
        #                                "Pdyn", "RmsBob", "Na_nl", "Np", "Np_nl"], "2h")
    
    
class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # self.clf = LogisticRegression(max_iter=1000)
        self.clf = HistGradientBoostingClassifier(max_iter=2000, max_leaf_nodes=15, l2_regularization=5e4, min_samples_leaf=150)
        # self.clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
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