import tensorflow as tf
import numpy as np

x = np.array(np.arange(10), dtype=float)
y = np.array(2 * np.arange(10), dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x, y, epochs=500)

print(model.predict([20.0]))
