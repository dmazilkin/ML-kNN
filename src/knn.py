import pandas as pd
import numpy as np
from typing import Tuple, List

class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self._available_metrics = {
            'euclidean': self.calc_euclidean,
            'manhattan': self.calc_manhattan,
            'chebyshev': self.calc_chebyshev,
            'cosine': self.calc_cosine,
        }
        
    def __str__(self):
        return f"MyKNNClf class: k={self.k}"
    
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
    
    def _predict_class(self, X_input: np.array) -> int:
        dist = self._available_metrics[self.metric](X_input)
        nearest = dist.sort_values(by='dist')[:self.k]
        nearest_classes = self.y[nearest.index]
        
        return 1 if nearest_classes[nearest_classes == 1].size >= nearest_classes[nearest_classes == 0].size else 0
        
    def _predict_class_proba(self, X_input: np.array) -> float:
        dist = self._available_metrics[self.metric](X_input)
        nearest = dist.sort_values(by='dist')[:self.k]
        nearest_classes = self.y[nearest.index]
        
        return nearest_classes.sum() / nearest_classes.size
    
    def predict(self, X_input: pd.DataFrame) -> pd.Series:
        y_predicted = np.zeros(X_input.shape[0], dtype='int')
        X_input.reset_index(inplace=True, drop=True)
        
        for feature in X_input.iterrows():
            ind = feature[0]
            data = feature[1].to_numpy().reshape((1, X_input.shape[1]))
            y_predicted[ind] = self._predict_class(data)
        
        return y_predicted
    
    def predict_proba(self, X_input: pd.DataFrame) -> List[float]:
        y_proba = np.zeros(X_input.shape[0], dtype='float')
        X_input.reset_index(inplace=True, drop=True)
        
        for feature in X_input.iterrows():
            ind = feature[0]
            data = feature[1].to_numpy().reshape((1, X_input.shape[1]))
            y_proba[ind] = self._predict_class_proba(data)
            
        return y_proba