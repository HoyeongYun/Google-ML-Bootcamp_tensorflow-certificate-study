import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Fashion MNIST 로드하는 방법
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# MNIST Fashion Data가 어떻게 생겼나
index = 0
np.set_printoptions(linewidth=320) # 한 line에 표시되는 character 수 지정
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY : \n {training_images[index]}')
plt.imshow(training_images[index])

# pixel 값 normalizing
training_images = training_images / 255.0
test_images = test_images / 255.0

# building classification model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Classification Model 의 최종 layer에서 Class가 선택되는 과정을 보여주는 예시
inputs = np.array([[1.0, 2.0, 3.0, 4.0, 2.0]])              # input을 shape(x,)로 주면 오륲
inputs = tf.convert_to_tensor(inputs)                       # ndarray를 tensor로 변환
print(f'input to softmax function : {inputs.numpy()}')      # 출력시에는 numpy object로 출력

outputs = tf.keras.activations.softmax(inputs)              # activations.softmax에 input을 넣으면
print(f'output of softmax fuction : {outputs.numpy()}')     # 어떻게 될까

sum = tf.reduce_sum(outputs)                                # 당연히 softmax의 output의 총합은 1
print(f'sum of outputs : {sum}')

prediction = np.argmax(outputs)
print(f'class with highest probability : {prediction}')    # output vector의 가장 높은 확률값이 계산된 class -> 모델로 예측한 class


# model compile, fitting
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)

# evaluate model
model.evaluate(test_images, test_labels)   # model.evaluate : [0.34194064140319824, 0.8758000135421753]

## Exercise 1
'''
predict 에 X를 주면 예측값 (여기서는 softmax를 통과한 값) 이 나옴
'''
classifications = model.predict(test_images) # Returns -> the probability that this item is each of the 10 classes
print(f'test_images.shape : {test_images.shape}')   # (10000, 28, 28)
print(classifications[0])   # test image 1000장 중 첫번째 이미지의 class 예측값
# print(classifications.shape) # (10000, 10)
print(test_labels[0]) # 9

## Exercise 2
'''
hidden layer 의 activation을 1024로 늘렸다.
by adding more Neurons we have to do more calculations, slowing down the process, but in this case they have a good impact -- we do get more accurate. That doesn't mean it's always a case of 'more is better', you can hit the law of diminishing returns very quickly!
'''
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print('model이 예측한 값')
print(np.argmax(classifications[0]))
print('실제값')
print(test_labels[0])