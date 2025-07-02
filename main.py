import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

from src.knn import MyKNNClf

N = 500

def create_data() -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    
    X = pd.DataFrame(np.zeros((N, 2)) + 10 * np.random.rand(N, 2))
    Y = (X.loc[:, 0] > X.loc[:, 1] - 2).astype(int)
    
    return X, Y

def main():
    X, Y = create_data()
    model = MyKNNClf(k=10)
    print(model)
    
if __name__ == '__main__':
    main()