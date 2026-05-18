from scipy.stats import zscore
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class StandardScaler(BaseEstimator, TransformerMixin):
    """Scale the data to given mean and std.

    Parameters
    ----------
    mean : str
        Rereferencing method.
    standard_deviation : int
        std
    """

    def __init__(self, mean=None, std=None, axis=0):
        self.mean = mean
        self.std = std
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return (X - X.mean(keepdims=True)) / (X.std(keepdims=True)
