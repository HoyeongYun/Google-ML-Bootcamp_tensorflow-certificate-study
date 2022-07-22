import wget
import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



if not os.path.exists('./sarcasm.json'):
    url = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'
    wget.download(url)

## json 파일 다루기

with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)            # json파일을 python객체로 변환

# print(type(datastore))                  # list
# print(len(datastore))                   # 26709

print(datastore[0])
print(datastore[20000])

# json.loads , json.dumps 문자열
# json.load, json.dump     파일

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(f'number of words in word_index : {len(word_index)}')
print(f'word_index : {word_index}')
print()

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

index = 2
print(f'sample headline : {sequences[index]}')              # list
print(f'padded sequence : {padded[index, :]}')              # ndarray  ,  ndarray[index, :] 은 ndarray[index] 와 같다
print()
print(f'shape of padded sequences : {padded.shape}')