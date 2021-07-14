import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.saving.saving_utils import model_input_signature

df = pd.read_csv("card_prices.csv", header=None)

dates = []
prices = []

for date in df[0]:
    dates.append(date)

for price in df[1]:
    prices.append(price)

training_data = {'Price': prices[1:2500],
        'Date': dates[1:2500]}

training_ds = pd.DataFrame(training_data).set_index('Date')

test_data = {'Price': prices[2500:-1],
        'Date': dates[2500:-1]}

test_ds = pd.DataFrame(test_data).set_index('Date')


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(training_ds.values.reshape(-1, 1))

prediction_days = 90

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days : x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

actual_prices = test_ds.values

total_dataset = pd.concat((training_ds, test_ds), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_ds)- prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color = "black", label="Actual Temple Garden Price")
plt.plot(predicted_prices, color = "green", label="Predicted Temple Garden Price")

plt.title("Temple Garden Price")
plt.xlabel("Time")
plt.ylabel("Temple Garden Price")
plt.legend()
plt.show()

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))


prediction = model.predict(real_data)
prediciton = scaler.inverse_transform(prediction)

print(f"Predicted Price: {prediciton}")