import wget
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from google.colab import files
from keras.preprocessing import image
import random

# wget 모듈을 이용해서 파일 다운로드하는 방법
url = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip'
wget.download(url)      # 현재 디렉토리에 다운받아짐

# zip 파일 해제 하는 방법
local_zip = './horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')           # ZipFile 객체 생성
zip_ref.extractall('./horse-or-human')              # extractall(path) : 모든파일 압축해제 | 지정한 path 아래에 풀기
zip_ref.close()

# horse, human image 개수를 세는 방법 (디렉토리 이름으로 os.listdir(해당디렉토리 path) -> 그 디렉토리 내 파일 이름이 list로 모두 담김 -> 그 list 의 length)
train_horse_dir = os.path.join('./horse-or-human/horses')
train_human_dir = os.path.join('./horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)                        # path로 넘겨준 directory 내에있는 파일명의 이름들을 list로 반환
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horses images : ', len(os.listdir(train_horse_dir)))
print('total training humans images : ', len(os.listdir(train_human_dir)))

# manual 하게 몇개의 이미지 보자
nrows = 4
ncols = 4

pic_index = 8

fig = plt.gcf()             # pyplot figure stack 이라는 게 있다는 거 알게됨
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index - 8 : pic_index]] # batch 마다의 이미지 path 생성
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index - 8 : pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')                          # The axes of the subplot 을 return

    img = mpimg.imread(img_path)            #mpimg.imread() path를 넣어주면 img array로 반환
    plt.imshow(img)

plt.show()

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

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

# Image Data Processing 하는 법
train_datagen = ImageDataGenerator(rescale=1/255)       # 데이터 제너레이터 객체 생성

train_generator = train_datagen.flow_from_directory(
    './horse-or-human',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1)                    # x, y 로 안넘기고 generator를 넘기네

# 내가 올린 이미지로 예측하기
uploaded = files.upload()         # 이 google api를 호출하면 내가 local에서 업로드할 이미지를 선택해주고 uploaded 에는 {업로드한 이미지 이름 : 실제 이미지파일} 이렇게 담긴다

for fn in uploaded.keys():        # 사용자가 이미지를 여러장 선택했을 때 모든 사진에 대해서 처리해줌

    path = './' + fn             # 내가 local에서 선택한 이미지를 올릴 곳
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)         # (h, w, c)
    x = np.expand_dims(x, axis=0)       # (1, h, w, c) 이렇게 바꿔줌

    images = np.vstack([x])                             # np.vstack(a), np.vstack([a]) 결과는 같음
    classes = model.predict(images, batch_size=10)      # predict할때 batch_size가 왜 필요하지?
    print(classes[0])       # human으로 예측할 확률

    if classes[0] > 0.5 :
        print(fn + ' is a human')
    else:
        print(fn + ' is a horse')

# Visualizing
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)                     # random 모듈 random.choice(list) 임의로 하나 선택해줌

img = load_img(img_path, target_size=(300, 300))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)   # x = x.reshape((1, ) + x.shape)

x /= 255

successive_feature_maps = visualization_model.predict(x)        # [두번째 layer에 대한 output, 세번째 layer에 대한 output, ...]

layer_names = [layer.name for layer in model.layers[1:]]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):

    if len(feature_map.shpae) == 4:     # output이 img인 것들

        n_features = feature_map.shape[-1]  # intermediate output의 채널 수(feature 수)
        size = feature_map.shape[1] # (1, size, size, n_features)
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
        plt.imshow(display_grid, aspect='auto', cmap='viridis')