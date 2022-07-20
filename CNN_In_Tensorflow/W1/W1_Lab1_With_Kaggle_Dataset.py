import matplotlib.image as mpimg
import wget
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import random


url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
wget.download(url)

local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()

zip_ref.close()

base_dir = './cats_and_dogs_filtered'

print("Contents of base directory")
print(os.listdir(base_dir))

print("\nContents of train directory")
print(os.listdir(os.path.join(base_dir, 'train')))

print("\nContents of validation directory")
print(os.listdir(os.path.join(base_dir, 'validation')))

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
validation_cat_fnames = os.listdir(validation_cats_dir)
validation_dog_fnames = os.listdir(validation_dogs_dir)

print("\ntrain cats names")
print(train_cat_fnames[:10])

print("\ntrain dogs names")
print(train_dog_fnames[:10])

print('\ntotal training cat images :', len(train_cat_fnames))
print('\ntotal training dog images :', len(train_dog_fnames))
print('\ntotal validation cat images : ', len(validation_cat_fnames))
print('\ntotal validation dogs images :', len(validation_dog_fnames))

nrows = 4
ncols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(nrows * 4, ncols * 4)

pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir, pic_fname) for pic_fname in train_cat_fnames[pic_index - 8 : pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, pic_fname) for pic_fname in train_dog_fnames[pic_index - 8 : pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):

    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

history = model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=15,
            validation_data=validation_generator,
            validation_steps=50,
            verbose=2
            )


# from google.colab import files
# from keras.preprocessing import image
#
# uploaded = files.upload()
#
# for fn in uploaded.keys():
#
#     path = './' + fn
#     img = image.load_img(path, target_size=(150, 150))
#
#     x = image.img_to_array(img)
#     x /= 255.0
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#
#     classes = model.predict(images, batch_size=10)
#
#     print(classes[0])
#
#     if classes[0] > 0.5:
#         print(fn + 'is a dog')
#     else:
#         print(fn + 'is a cat')

succesive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=succesive_outputs)

cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x /= 255.0

succesive_feature_map = visualization_model.predict(x)          #[첫번째 까지만 통과한 결과, 두번째 까지만 통과한 결과]

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, succesive_feature_map):
    # feature_map  (1, size, size, channels(features))
    if len(feature_map.shape) == 4:

        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]

        display_grid = np.zeros((size, size * n_features))

        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x

        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmp='viridis')

acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )