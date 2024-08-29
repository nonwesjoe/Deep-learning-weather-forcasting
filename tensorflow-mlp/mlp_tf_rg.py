import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

epochs=100
batch_size=32
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

model=Sequential([
    Dense(8,activation='relu'),
    Dense(5,activation='relu'),
    Dense(5,activation='relu'),
    Dense(1)
])

model.summary()
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.fit(X_train,y_train,
          batch_size=32,
          epochs=100,
          validation_data=(X_test,y_test)
          )

model.save('mlp_tf_rg.keras')
pre=model.predict(X_test)


plt.plot( y_test[500:1500], label='val')
plt.plot(pre[500:1500], label='pre')
plt.legend()
plt.show()
