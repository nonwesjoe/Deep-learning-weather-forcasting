import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model=load_model('mlp_tf_rg.keras')

data = pd.read_csv('vali.csv')
X = data[['p (mbar)', 'rh (%)']].values
Y = data['T (degC)'].values

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y.reshape(-1,1))

pre=model.predict(X)

print(pre.shape)
print(Y.shape)

y_pre= scaler_Y.inverse_transform(pre)
y_val= scaler_Y.inverse_transform(Y)

plt.plot(y_pre,label='pre')
plt.plot(y_val,label='val')
plt.show()