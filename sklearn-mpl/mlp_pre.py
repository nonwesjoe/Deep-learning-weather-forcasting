import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

model = joblib.load('mlp.pkl')
# 读取CSV文件
data = pd.read_csv('vali.csv')
X = data[['p (mbar)', 'rh (%)']].values
Y = data['T (degC)'].values
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y.reshape(-1,1))

pre=model.predict(X)
val=Y

print(pre.shape)
print(val.shape)
# 在测试集上进行预测

# 计算并打印均方误差
mse = mean_squared_error(pre, val)
print("Test Mean Squared Error:", mse)

# 可视化预测结果
plt.plot(pre, label='pre')
plt.plot(val, label='val')
plt.legend()
plt.show()
