import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

from src.knn_classification import MyKNNClf

N = 50
N_test = 10

def create_data_classification() -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X.loc[:, 0] > X.loc[:, 1] - 2).astype(int)
    
    return X, Y

def create_test_data_classification() -> pd.DataFrame:
    return pd.DataFrame(np.zeros((N_test, 2)) + 10 * np.random.rand(N_test, 2))

def classification_example():
    X, y = create_data_classification()
    model = MyKNNClf(k=10, metric='cosine', weight='rank')
    size = model.fit(X, y)
    
    X_test = create_test_data_classification()
    y_test = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    figure, axis = plt.subplots(1, 2)
    axis[0].scatter(X.to_numpy()[y == 1][:, 0], X.to_numpy()[y == 1][:, 1], color='blue')
    axis[0].scatter(X.to_numpy()[y == 0][:, 0], X.to_numpy()[y == 0][:, 1], color='red')
    axis[0].set_title('Real data')
    
    axis[1].scatter(X_test.to_numpy()[y_test == 1][:, 0], X_test.to_numpy()[y_test == 1][:, 1], color='blue')
    axis[1].scatter(X_test.to_numpy()[y_test == 0][:, 0], X_test.to_numpy()[y_test == 0][:, 1], color='red')
    axis[1].set_title('Predicted data')
    plt.show()