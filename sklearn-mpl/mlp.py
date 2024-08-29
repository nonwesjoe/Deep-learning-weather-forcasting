import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
# 读取CSV文件
data = pd.read_csv('weather.csv')
X = data[['p (mbar)', 'rh (%)']].values
Y = data['T (degC)'].values

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y.reshape(-1,1))

X_train=np.array(X[0:40000])
y_train=np.array(Y[0:40000])
X_test=np.array(X[40000:50000])
y_test=np.array(Y[40000:50000])


# 初始化并训练神经网络回归器
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(30,20),
    max_iter=30,
    alpha=0.0001,
    solver='adam',
    random_state=42,
    verbose=True
)

# 训练模型
mlp_regressor.fit(X_train, y_train)
joblib.dump(mlp_regressor, 'mlp.pkl')
# 在测试集上进行预测
y_pred = mlp_regressor.predict(X_test)

# 计算并打印均方误差
mse = mean_squared_error(y_test, y_pred)
print("Test Mean Squared Error:", mse)

# 可视化预测结果
plt.plot(y_test[500:1000], label='Actual Temperature')
plt.plot(y_pred[500:1000], label='Predicted Temperature')
plt.legend()
plt.show()
