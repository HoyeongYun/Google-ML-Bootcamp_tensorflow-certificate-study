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

'''
Exercise 몇개 skip
'''

print(training_images.shape)
default_batch_size = 32
print(divmod(training_images.shape[0], default_batch_size)) # 이래서 epoch 당 1875번 반복했던 거

# 우리가 원하는 accuracy에 도달하면 더 반복하지 않고 끝내는 방법
# Callback 사용법
class myCallback(tf.keras.callbacks.Callback):      # tf.callbacks.Callback에는 각 epoch 뿐 아니라 각 mini batch 등 여러 순간에 호출되는 method를 포함하므로 적절히 오버라이딩 하여 사용하면 될듯
    def on_epoch_end(self, epoch, logs={}):         # epoch 끝날때 마다 반복해서 불리는 method overriding 하는 듯 -- 실제로 불렀을 땐 epoch 끝나가 조금 전에 불리는 듯
        if(logs.get('accuracy') >= 0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # metrics 내에 어떤 평가지표로 평가할지를 넣어줌 이는 epoch 마다 trainig process를 모니터링 할때 필요
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])                    # callbacks=[] 안에는 자신의 custom callback 객체를 넣어줌