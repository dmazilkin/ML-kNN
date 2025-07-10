import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

from src.knn_regression import MyKNNReg

N = 50
N_test = 10

def create_data_regression() -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = 5 + 2 * X.loc[:, 0] + X.loc[:, 1] ** 2
    
    return X, Y

def create_test_data_regression() -> pd.DataFrame:
    return pd.DataFrame(np.zeros((N_test, 2)) + 10 * np.random.rand(N_test, 2))

def regression_example():
    X, y = create_data_regression()
    model = MyKNNReg(k=10, weight='distance')
    size = model.fit(X, y)
    
    X_test = create_test_data_regression()
    y_test = model.predict(X_test)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.scatter3D(X.to_numpy()[:, 0], X.to_numpy()[:, 1], y.to_numpy(), color='blue')
    ax.scatter3D(X_test.to_numpy()[:, 0], X_test.to_numpy()[:, 1], y_test, color='green')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')

    plt.show()