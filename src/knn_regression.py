import pandas as pd
import numpy as np
from typing import Tuple

class MyKNNReg:

    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.metric = metric
        self._available_metrics = {
            'euclidean': self.calc_euclidean,
            'manhattan': self.calc_manhattan,
            'chebyshev': self.calc_chebyshev,
            'cosine': self.calc_cosine,
        }
        self.weight = weight
        self._available_weights = {
            'uniform': self.predict_by_count,
            'distance': self.predict_by_distance,
            'rank': self.predict_by_rank,
        }
        
    def __str__(self):
        return f"MyKNNReg class: k={self.k}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[int, int]:
        self.X = X
        self.y = y
        self.train_size = X.shape
        
        return self.train_size
    
    def calc_euclidean(self, X_input: np.array) -> pd.DataFrame:
        return pd.DataFrame({'dist': np.linalg.norm(X_input - self.X.to_numpy(), ord=2, axis=1)})
    
    def calc_chebyshev(self, X_input: np.array) -> pd.DataFrame:
        return pd.DataFrame({'dist': np.max(np.abs(X_input - self.X.to_numpy()), axis=1)})
    
    def calc_manhattan(self, X_input: np.array) -> pd.DataFrame:
        return pd.DataFrame({'dist': np.linalg.norm(X_input - self.X.to_numpy(), ord=1, axis=1)})
    
    def calc_cosine(self, X_input: np.array) -> pd.DataFrame:
        return pd.DataFrame({'dist': 1 - np.sum(self.X * X_input, axis=1) / np.linalg.norm(self.X, ord=2, axis=1) / np.linalg.norm(X_input, ord=2, axis=1)})
    
    def predict_by_count(self, nearest: pd.DataFrame) -> float:
        nearest_target = self.y[nearest.index]
        
        return np.mean(nearest_target)
    
    def predict_by_distance(self, nearest: pd.DataFrame) -> float:
        nearest_target = self.y[nearest.index]
        weights = 1 / nearest['dist'] / np.sum(1 / nearest['dist'])
        
        return np.sum(nearest_target * weights)
    
    def predict_by_rank(self, nearest: pd.DataFrame) -> float:
        nearest_target = self.y[nearest.index]
        ind = nearest_target.reset_index(drop=True).index + 1
        weights = 1 / ind / np.sum(1 / ind)
        
        return np.sum(nearest_target * weights)
    
    def _predict_target(self, X_input: np.array) -> float:
        dist: pd.DataFrame = self._available_metrics[self.metric](X_input)
        nearest: pd.DataFrame = dist.sort_values(by='dist')[:self.k]
        target: float = self._available_weights[self.weight](nearest)

        return target
    
    def predict(self, X_input: pd.DataFrame) -> pd.Series:
        y_predicted = np.zeros(X_input.shape[0], dtype='float')
        X_input.reset_index(inplace=True, drop=True)
        
        for feature in X_input.iterrows():
            ind = feature[0]
            data = feature[1].to_numpy().reshape((1, X_input.shape[1]))
            y_predicted[ind] = self._predict_target(data)
        
        return y_predicted