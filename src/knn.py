import pandas as pd
import numpy as np
from typing import Tuple, List

class MyKNNClf:
    def __init__(self, k: int = 3):
        self.k = k
    
    def __str__(self):
        return f"MyKNNClf class: k={self.k}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Tuple[int, int]:
        self.X = X
        self.y = y
        self.train_size = X.shape
        
        return self.train_size
    
    def _predict_class(self, X_input: np.array) -> int:
        dist = pd.DataFrame({'l2': np.linalg.norm(X_input - self.X.to_numpy(), ord=2, axis=1)})
        nearest = dist.sort_values(by='l2')[:self.k]
        nearest_classes = self.y[nearest.index]
        
        return 1 if nearest_classes[nearest_classes == 1].size >= nearest_classes[nearest_classes == 0].size else 0
        
    def _predict_class_proba(self, X_input: np.array) -> float:
        dist = pd.DataFrame({'l2': np.linalg.norm(X_input - self.X.to_numpy(), ord=2, axis=1)})
        nearest = dist.sort_values(by='l2')[:self.k]
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