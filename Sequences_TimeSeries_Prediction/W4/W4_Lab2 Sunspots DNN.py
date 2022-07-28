import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import wget
import os

# utility functions
def plot_series(x, y, format="-", start=0, end=None, title=None, xlabel=None, ylabel=None, legend=None):

    plt.figure(figsize=(10, 6))

    if type(y) is tuple:
        for y_curr in y:
            plt.plot(x[start:end], y_curr[start:end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)

    plt.title(title)
    plt.grid(True)
    plt.show()

# Download dataset and parsing
if not os.path.exists('./Sunspots.csv'):
    url = 'https://storage.googleapis.com/tensorflow-1-public/course4/Sunspots.csv'
    wget.download(url)

time_steps = []
sunspots = []

with open('./Sunspots.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    next(reader)

    for row in reader:
        time_steps.append(int(row[0]))
        sunspots.append(float(row[2]))

time = np.array(time_steps)
series = np.array(sunspots)

plot_series(time, series, xlabel='Month', ylabel='Monthly Mean Total Sunspot Number')

# split data
split_time = 3000

time_train = time[:split_time]
series_train = series[:split_time]

time_valid = time[split_time:]
series_valid = series[split_time:]

# data -> dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda t: (t[:-1], t[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

train_set = windowed_dataset(series_train, window_size, batch_size, shuffle_buffer_size)

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, input_shape=[window_size], activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.summary()

init_weights = model.get_weights()

# pre-training for searching optimial learning_rate (Tune the learning rate)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

model.complile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(momentum=0.9))

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# select lr
lr = 1e-8 * 10**(np.arange(100) / 20)

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.semilogx(lr, history.history['loss'])
plt.tick_params('both', length=10, width=1, which='both')
plt.axis([1e-8, 1e-3, 0, 100])
plt.show()

# main training
tf.keras.backend.clear_session()

model.set_weights(init_weights)

learning_rate = 2e-5

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizer.SGD(learning_rate=learning_rate, momentum=0.9), metrics=['mae'])

history = model.fit(train_set, epochs=100)

# model prediction
def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast

forecast_series = series[split_time - window_size : -1]

forecast = model_forecast(model, forecast_series, window_size, batch_size)

results = forecast.squeeze()

plot_series(time_valid, (series_valid, results))

print(tf.keras.metrics.mean_absolute_error(series_valid, results).numpy())






