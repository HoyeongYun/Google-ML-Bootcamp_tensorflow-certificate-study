import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import wget
import os

# utility
def plot_series(x, y, format='-', start=0, end=None, title=None, xlabel=None, ylabel=None, legend=None):
    plt.figure(figsize=(10, 6))

    if type(y) is tuple:
        for y_curr in y:
            plt.plot(x[start:end], y_curr[start:end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title(title)

    if legend:
        plt.legend(legend)

# sunspot data download
if not os.path.exists('./Sunspots.csv'):
    url = 'https://storage.googleapis.com/tensorflow-1-public/course4/Sunspots.csv'
    wget.download(url)

time_step = []
sunspots = []

# parsing data
with open('./Sunspots.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    next(reader)

    for row in reader:
        time_step.append(int(row[0]))
        sunspots.append(float(row[2]))

time = np.array(time_step)
series = np.array(sunspots)

plot_series(time, series, xlabel='Month', ylabel='Monthly Mean Total Sunspot Number')

# split data
split_time = 1000

time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

# making dataset Function & raw data -> dataset
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

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# build Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='causal', input_shape=[window_size, 1]),    # conv layer??? ????????? 3?????? ?????? input ??????, 'causal' ?????? time step size ??????
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

model.summary()

# Tune the Learning Rate, lr??? ??????????????? loss??? ??????????????? ?????? lr ????????? ????????? ????????? ??? rough?????? ??????????????? ??????
init_weights = model.get_weights()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * (10 ** (epoch/20)))

optimizer = tf.keras.optimizers.SGD(momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)

history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# plot results (lr vs loss)
lrs = 1e-8 * (10 ** (np.arange(100)/20))

plt.figure(figsize=(10, 6))

plt.semilogx(lrs, history.history['loss'])

plt.tick_params('both', width=1, legth=10, which='both')

plt.axis([1e-8, 1e-3, 0, 100])

plt.show()

# lr = 8e-7 ?????? lr??? model retraining, ?????? ????????? ????????? ???????????? ?????? save??? weight??????
tf.keras.backend.clear_session()

model.set_weigths(init_weights)

learning_rate = 8e-7

optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber, optimizer=optimizer, metrics=['mae'])

history = model.fit(train_set, epochs=100)

# plot mae vs loss -> ?????? ???????????? ??? ?????? -> ?????? ?????? range?????? ????????? ??????
mae = history.history['mae']
loss = history.history['loss']

epochs = range(len(loss))

plot_series(x=epochs, y=(mae, loss), title='MAE and Loss', xlabel='MAE', ylabel='Loss', legend=['MAE', 'Loss'])

zoom_split = int(epochs[-1] * 0.2)
epochs_zoom = epochs[zoom_split:]
mae_zoom = mae[zoom_split:]
loss_zoom = loss[zoom_split:]

plot_series(x=epochs_zoom, y=(mae_zoom, loss_zoom), title='MAE and Loss', xlabel='MAE', ylabel='Loss', legend=['MAE', 'Loss'])

# Model Prediction
def model_forecast(model, series, window_size, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)

    return forecast

forecast_series = series[split_time - window_size : -1]

forecast = model_forecast(model, forecast_series, window_size, batch_size)

results = forecast.squeeze()

plot_series(time_valid, (forecast, results))

print(tf.keras.metrics.mean_absolute_error(forecast, results).numpy())


# Plus Alpha - learning rate decay implementation
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

history = model.fit(train_set, epochs=10)

### vs ###

initial_learning_rate = 1e-7

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=400, decay_rate=0.96, staircase=True)   # tf.keras.callbacks.LearningRateScheduler ??? callback, ??????

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)        # ????????? ?????????

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

history = model.fit(train_set, epochs=40)