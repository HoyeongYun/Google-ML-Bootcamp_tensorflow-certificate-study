import keras.callbacks
import wget
import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load data
# train set
# wget.download('https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip')
# # test set
# wget.download('https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip')

# zip_ref = zipfile.ZipFile('./rps.zip', 'r')
# zip_ref.extractall()
# zip_ref = zipfile.ZipFile('./rps-test-set.zip', 'r')
# zip_ref.extractall()
# zip_ref.close()

train_dir = os.path.join('rps/')
val_dir = os.path.join('rps-test-set')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])          # label 이 one_hot encoding일 때 categorical_crossentropy

train_datagen = ImageDataGenerator(
      rescale=1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')) > 0.85 and (logs.get('val_accuracy')) > 0.8:
            print('\n stop training')
            self.model.stop_training = True

callbacks = myCallback()

history = model.fit(train_generator, validation_data=val_generator, epochs=25, callbacks=[callbacks])

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()