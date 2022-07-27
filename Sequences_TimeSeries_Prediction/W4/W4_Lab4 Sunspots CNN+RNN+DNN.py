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
if not os.exist('./Sunspots.csv'):
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
