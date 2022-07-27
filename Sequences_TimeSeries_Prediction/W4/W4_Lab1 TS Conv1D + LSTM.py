import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# utility functions
def plot_series(x, y, format='-', start=0, end=None, title=None, xlabel=None, ylabel=None, legend=None):

    plt.figure(figsize=(10, 6))

    if type(y) is tuple:
        for y_curr in y:
            plt.plot(x[start:end], y_curr[start:end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)

    if legend:
        plt.legend(legend)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def trend(time, slope=0):
    series = time * slope
    return series

def seasonal_pattern(season_time):
    data_pattern = np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))
    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    data_pattern = amplitude * seasonal_pattern(season_time)
    return data_pattern

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    noise = rnd.random(len(time)) * noise_level
    return noise

# generate synthetic data
time = np.arange(365 * 4 + 1, dtype='float32')
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, 365, amplitude)
series += noise(time, noise_level, seed=42)

plot_series(time, series, xlabel='Time', ylabel='Value')

# split dataset
split_time = 1000

time_train = time[:split_time]
series_train = series[:split_time]

time_valid = time[split_time:]
series_train = series[split_time:]

# series data -> dataset -> train_dataset
window_size = 20
batch_size = 16
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda t: (t[:-1], t[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_dataset = windowed_dataset(series_train, window_size, batch_size, shuffle_buffer_size)

# build model || Conv1D -> LSTM -> LSTM -> Dense -> *400
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu', input_shape=[window_size, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1),
    tf.keras.layer.Lambda(lambda x: x * 400)
])

model.summary()












