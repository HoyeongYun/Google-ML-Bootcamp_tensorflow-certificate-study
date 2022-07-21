import zipfile
import wget
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# url1 = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip'
# url2 = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip'
# wget.download(url1)
# wget.download(url2)
#
# zip_ref = zipfile.ZipFile('./horse-or-human.zip', 'r')
# zip_ref.extractall('./horse-or-human')
#
# zip_ref = zipfile.ZipFile('./validation-horse-or-human.zip', 'r')
# zip_ref.extractall('./validation-horse-or-human')
#
# zip_ref.close()

train_dir = os.path.join('./horse-or-human')
train_horse_dir = os.path.join(train_dir, 'horses')
train_human_dir = os.path.join(train_dir, 'humans')

validation_dir = os.path.join('./validation-horse-or-human')
validation_horse_dir = os.path.join(validation_dir, 'horses')
validation_human_dir = os.path.join(validation_dir, 'humans')

print('Training Set 크기 :', len(os.listdir(train_horse_dir) + os.listdir(train_human_dir)))
print('Validation Set 크기 :', len(os.listdir(validation_horse_dir) + os.listdir(validation_human_dir)))

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=10,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    batch_size=10,
    class_mode='binary'
)

EPOCHS = 20

history = model.fit(
    train_generator,
    steps_per_epoch=103,                            # len(train size(전체 데이터 크기)) // (train batch size) 보다 큰 값을 넣으면 오류가 난다
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=26,
    verbose=1
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


