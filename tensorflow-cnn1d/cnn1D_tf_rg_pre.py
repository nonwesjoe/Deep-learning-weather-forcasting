import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model=load_model('cnn1D_tf_rg.keras')
data = pd.read_csv('vali.csv')
X = data[['p (mbar)', 'rh (%)']].values
Y = data['T (degC)'].values

x, y = [], []
for i in range(6, len(X)):
    x.append(X[i - 6:i])
    y.append(Y[i])

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

pre=model.predict(x)
val=Y

print(pre.shape)
print(val.shape)
plt.plot(pre,label='pre')
plt.plot(val,label='val')
plt.show()