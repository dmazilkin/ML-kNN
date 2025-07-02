import pandas as pd

class MyKNNClf:
    def __init__(self, k: int = 3):
        self.k = k
    
    def __str__(self):
        return f"MyKNNClf class: k={self.k}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape
        
        return self.train_size
