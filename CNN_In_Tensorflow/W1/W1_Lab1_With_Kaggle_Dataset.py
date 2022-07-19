import matplotlib.image as mpimg
import wget
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


# url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# wget.download(url)

# local_zip = './cats_and_dogs_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall()
#
# zip_ref.close()

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
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

#뒤에 남음


