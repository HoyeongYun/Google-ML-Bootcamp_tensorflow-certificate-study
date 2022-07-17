import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Load dataset and preprocessing
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

# Convolution 사용하지 않는 Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('\nMODEL TRAINIG')
model.fit(training_images, training_labels, epochs=5)

print('\nMODEL EVALUATION')
model.evaluate(test_images, test_labels)

# Convolution Model
model = tf.keras.models.Sequential([
    # Feature Extraction
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Classification
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# layer, output shape, params 등을 출력
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('\n MODEL TRAINING')
model.fit(training_images, training_labels, epochs=5)

print('\nMODEL EVALUATION')
model.evaluate(test_images, test_labels)

# Visualizing Convolutions and Pooling
# 각 layer 마다의 결과를 Visualizing 하는 방법
print(test_labels[:100])

import matplotlib.pyplot as plt
from tensorflow.keras import models
import numpy as np

FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 1

# 아직 Functional API 익숙지 않지만, 이렇게 outputs으로 output layer 하나가 아니라 [layer1 ouput, layer2 ouput, ... ] 리스트를 넘기면
# predict를 했을때 [layer1만 통과한 output, layer2까지 통과한 output, ...] 으로 나오는 듯
# 기존에는 마지막 레이어가 softmax였기 때문에 [category 1에 속할 확률, category 2에 속할 확률, ..] 만 나옴
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# print(layer_outputs)
# np.set_printoptions(linewidth=320)
# print(test_images[0].shape)  # shape 28, 28
# print(test_images[0].reshape(1, 28, 28, 1))
# print(activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1)))
# print(activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[0])
# print(activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[0].shape)

plt.figure()
plt.imshow(test_images[FIRST_IMAGE])
plt.figure()
plt.imshow(test_images[SECOND_IMAGE])
plt.figure()
plt.imshow(test_images[THIRD_IMAGE])

f, axarr = plt.subplots(3, 4)

for x in range(0, 4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]      # predict에는 총 batch input (m, h, w, c)를 넘겨야하므로 reshape
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')                   # [x] -> (x+1)번째 layer에 통과시킨 결과
    axarr[0, x].grid(False)                                                               # CONVOLUTION_NUMBER => 이 case에서는 32개 채널 중 몇번 째 channel인지

    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)