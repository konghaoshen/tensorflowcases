#梯度下降预测房价 demo

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing(data_home='C:/Users/Shinelon/scikit_learn_data',download_if_missing=True)
m, n = housing.data.shape
print(m, n)
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing]
scaler = StandardScaler.fit(housing_data_plus_bias)
scaler_horsing_data_plus_bias = scaler.transform(housing_data_plus_bias)

