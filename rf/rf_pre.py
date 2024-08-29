import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
model=joblib.load('rf.pkl')
# 读取数据
data = pd.read_csv('vali.csv')

x = data[['p (mbar)', 'rh (%)']].values
y = data['T (degC)'].values
# 数据归一化
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
x = scaler_X.fit_transform(x)
y = scaler_Y.fit_transform(y.reshape(-1,1))


y_pred = model.predict(x)
y_val=y


# 评估模型
mse = mean_squared_error(y_val, y_pred)
print(f"均方误差: {mse}")

plt.plot(y_pred,label='y-pred')
plt.plot(y_val ,label='y-test')
plt.show()

