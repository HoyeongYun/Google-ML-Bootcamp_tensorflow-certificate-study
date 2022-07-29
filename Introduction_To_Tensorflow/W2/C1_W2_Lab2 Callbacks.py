import tensorflow as tf

# data load, preprocess
fmnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fmnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# Creating a Callback custom class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print('\nLoss is lower than 0.4 so cacelling training!')
            self.model.stop_training = True

callbacks = myCallback()

# Define and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # training process accuracy도 출력함

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])