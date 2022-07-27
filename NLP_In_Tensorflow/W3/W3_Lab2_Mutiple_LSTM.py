import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

tokenizer = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 32

train_data, test_data = dataset['train'], dataset['test']

train_dataset = train_data.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

batch_size = 1
timesteps = 20
features = 16
lstm_dim = 8

print(f'batch_size: {batch_size}')
print(f'timesteps (sequence length): {timesteps}')
print(f'features (embedding size): {features}')
print(f'lstm output units: {lstm_dim}')

random_input = np.random.rand(batch_size, timesteps, features)
print(f'shape of input array : {random_input.shape}')               # (1, 20, 16)

lstm = tf.keras.layers.LSTM(lstm_dim)
result = lstm(random_input)
print(f'shape of lstm output(return_sequences=False) : {result.shape}')         # (1, 8)

lstm_rs = tf.keras.layers.LSTM(lstm_dim, return_sequences=True)
result = lstm_rs(random_input)
print(f'shape of lstm output(return_sequences=True) : {result.shape}')          # (1, 20, 8)

# https://medium.com/deep-learning-with-keras/lstm-understanding-output-types-e93d2fb57c77 보고 이해
# 결국 lstm_dim 이라는게 hidden unit 하나에 할당되는 차원수라는 거
# lstm_unit이라는 말을 써서 내가아는 lstm 유닛개수가 인풋의 time series 와 다른 건가? 하고 헷갈렸지만 당연히 그 둘은 같다
# 대신, return_sequence=False 즉 default 로 lstm_dim만 넘겼을 때, 마지막 유닛을 통과하고 나온 hidden state를 배출하는데
# 이 hidden state를 lstm_dim 차원 만큼으로 내보낸다는 뜻 ( input으로 한 time step 당 (한개단어 당) features만큼 의 차원이
# 통과한 후에는 litm_dim 만큼 변경돼서 나간다고 생각하면 된다.

embedding_dim = 64
lstm1_dim = 64
lstm2_dim = 32
dense_dim = 64

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),
    tf.keras.layers.Dense(dense_dim, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("######################################################################")
my_input = np.random.rand(1, 5, 16)
print(f'input shape : {my_input.shape}')
embedding_layer = tf.keras.layers.Embedding(100, 16)
output_of_embedding = embedding_layer(my_input)
print(f'first output shape : {output_of_embedding.shape}')
lstm_layer1 = tf.keras.layers.LSTM

#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# NUM_EPOCHS = 10
#
# # Train the model
# history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)
#
# def plot_graphs(history, string):
#   plt.plot(history.history[string])
#   plt.plot(history.history['val_'+string])
#   plt.xlabel("Epochs")
#   plt.ylabel(string)
#   plt.legend([string, 'val_'+string])
#   plt.show()
#
# # Plot the accuracy and results
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")