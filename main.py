import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# doc file
data = pd.read_csv(r'D:\duancoban\MLbase\weather_forecast_data.csv')
# print(data.head())
# data.info()

# chia data
N, d = data.shape
x = data.iloc[:, 0:d-1]
y = data.iloc[:, d-1].values.reshape(-1,1)
# print(x)
# print(y)



