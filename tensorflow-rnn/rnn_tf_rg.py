import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# tensorflow 2.17.0, keras 3.4.1
# 读取数据
data = pd.read_csv('weather.csv')

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
x_train=np.array(x[0:35000])
y_train=np.array(y[0:35000])
x_test=np.array(x[35000:50000])
y_test=np.array(y[35000:50000])

# 构建LSTM模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(win, x.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
model.save('rnn_tf_rg.keras')


# 评估模型
y_pred = model.predict(x_test)
# 可视化预测结果
plt.plot(y_test[1000:2000], label='Actual Temperature')
plt.plot(y_pred[1000:2000], label='Predicted Temperature')
plt.legend()
plt.show()
