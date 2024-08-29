import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
# Load and prepare your data
data = pd.read_csv('weather.csv')
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

xtrain=x[0:40000]
ytrain=y[0:40000]
xtest=x[40000:50000]
ytest=y[40000:50000]
# Split the data into training and testing sets

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=2, activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Flatten())
model.add(Dense(1, activation='linear')) # Output layer for predicting 7 days

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model
model.fit(xtrain, ytrain, epochs=20, batch_size=64, validation_data=(xtest, ytest))
model.save('cnn1D_tf_rg.keras')


pre=model.predict(xtest[1000:2000])
val=ytest[1000:2000]


print(pre.shape)
print(val.shape)
plt.plot(pre,label='pre')
plt.plot(val,label='val')
plt.show()