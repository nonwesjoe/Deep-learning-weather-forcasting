import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('rnn_tf_rg.keras')
# 读取数据
data = pd.read_csv('vali.csv')
# 选择输入特征和目标值
X = data[['p (mbar)', 'rh (%)']].values
Y = data['T (degC)'].values
# 数据归一化
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y.reshape(-1,1))
# 设置时间步长
win = 6
x, y = [], []
for i in range(win, len(X)):
    x.append(X[i-win:i, :])
    y.append(Y[i])

x, y = np.array(x), np.array(y)

print(x.shape)
print(y.shape)
# 划分训练集和测试集
x_val=np.array(x[50000:])
y_val=np.array(y[50000:])

y_pre=model.predict(x_val)
print(y_pre.shape)
print(y_val.shape)

y_pre= scaler_Y.inverse_transform(y_pre)
y_val= scaler_Y.inverse_transform(y_val)
plt.figure()
plt.plot(y_pre,label="pre")
plt.plot(y_val,label="val")
plt.show()
