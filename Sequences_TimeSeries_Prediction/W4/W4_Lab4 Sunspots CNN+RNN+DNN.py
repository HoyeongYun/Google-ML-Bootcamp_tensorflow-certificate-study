import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

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

url = 'https://storage.googleapis.com/tensorflow-1-public/course4/Sunspots.csv'
wget.download(url)

time_step = []
sunspots = []

with open('./Sunspots.csv', 'r') as csvfile: