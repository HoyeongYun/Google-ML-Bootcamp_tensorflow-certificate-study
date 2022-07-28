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
series_valid = series[split_time:]

# series data -> dataset -> train_dataset
window_size = 20
batch_size = 16
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):          # validation set 만들때도 활용
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
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='relu', input_shape=[window_size, 1]),        # 알기로는, input_shape에는 내 dataset 의 (첫번째 dimension을 제외한, 즉b atch_size를 제외한)shape을 적어 줘야하는 것인 줄 알았는데
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1),
    tf.keras.layer.Lambda(lambda x: x * 400)
])

model.summary()

# Tune the Learning Rate | lr vs loss graph를 그리기 위해 lr 증가시켜가면서 training
init_weights = model.get_weights()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * (10 ** (epoch/20)))

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(momentum=0.9))

history = model.fit(train_dataset, epochs=100, callbacks=[lr_schedule])

# plot lr vs loss
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.semilogx(lrs, history.history['loss'])
plt.tick_params('both', length=10, width=1, which='both')
plt.axis([1e-8, 1e-3, 0, 50])

# train the model || 여기가 main training
tf.keras.backend.clear_session()

model.set_weights(init_weights)

learning_rate = 1e-7

model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate, momentum=0.9), metrics=['mae'])

history = model.fit(train_dataset, epochs=500)

# plot mae vs loss
mae = history.history['mae']
loss = history.history['loss']

epochs = range(len(loss))

plot_series(
    x=epochs,
    y=(mae, loss),
    title='MAE and Loss',
    xlabel='Epochs',
    legend=['MAE', 'Loss']
    )

zoom_split = int(epochs[-1] * 0.2)
epochs_zoom = epochs[zoom_split:]
mae_zoom = mae[zoom_split:]
loss_zoom = loss[zoom_split:]

plot_series(
    x=epochs_zoom,
    y=(mae_zoom, loss_zoom),
    title='MAE and Loss',
    xlabel='Epochs',
    legend=['MAE', 'Loss']
    )

# prediction
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

plot_series(time_valid, (series_valid, results))

print(tf.keras.metrics.mean_absolute_error(series_valid, results).numpy())
print(tf.keras.metrics.mean_squared_error(series_valid, results).numpy())

# Plus alpha | early stopping 복습
val_set = windowed_dataset(series_valid, window_size, batch_size, shuffle_buffer_size)

model.set_weights(init_weights)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_mae')) < 5.2:

            print('\n ~~~~')
            self.model.stop_running = True

callback = myCallback()

learning_rate = 4e-8

optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

history = model.fit(train_set, epochs=500, validation_data=val_set, callbacks=[callback])

