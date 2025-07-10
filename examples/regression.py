import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

from src.knn_regression import MyKNNReg

def create_data_regression(N: int) -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = 5 + 2 * X.loc[:, 0] + X.loc[:, 1] ** 2
    
    return X, Y

def create_test_data_regression(N_test: int) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((N_test, 2)) + 10 * np.random.rand(N_test, 2))

def regression_example(train: int, predict: int, k: int, weight: str, metric: str):
    N_train, N_test = train, predict
    X, y = create_data_regression(N_train)
    model = MyKNNReg(k=k, weight=weight, metric=metric)
    size = model.fit(X, y)
    
    X_test = create_test_data_regression(N_test)
    y_test = model.predict(X_test)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.scatter3D(X.to_numpy()[:, 0], X.to_numpy()[:, 1], y.to_numpy(), color='blue')
    ax.scatter3D(X_test.to_numpy()[:, 0], X_test.to_numpy()[:, 1], y_test, color='green')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')

    plt.show()