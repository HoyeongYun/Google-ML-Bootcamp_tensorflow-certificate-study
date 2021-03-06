import tensorflow as tf
import numpy as np

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# print(tf.test.is_built_with_cuda())
# print(tf.test.is_gpu_available())

x = np.array(np.arange(10), dtype=float)
y = np.array(2 * np.arange(10), dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x, y, epochs=500)

print(model.predict([20.0]))
