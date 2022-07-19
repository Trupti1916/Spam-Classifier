from keras.layers import LSTM, SimpleRNN, Embedding, Dense
from keras.models import Model, Sequential
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

data = pd.read_csv("spam.csv", encoding='latin-1')
print(data.head(5))

texts = []
labels = []
for i, label in enumerate(data['Category']):
    texts.append(data['Message'][i])
    if label == 'ham':
        labels.append(0)
    else:
        labels.append(1)        

texts = np.asarray(texts)        
labels = np.asarray(labels)

print("Number of Texts :", len(texts))
print("Number of Labels :", len(labels))

#number of words used as features
max_feature = 10000
maxLen = 500

#We will use 80% of data as Traning & 20% as Validation
training_samples = int(5572 * 0.8)
validation_samples = int(5572 - training_samples)

#Sanity Checks
print(len(texts) == (training_samples + validation_samples))
print("Number of Training= {0} , Validation= {1} ".format(training_samples, validation_samples))

tokens = Tokenizer()
tokens.fit_on_texts(texts)
sequences = tokens.texts_to_sequences(texts)

word_index = tokens.word_index
print("Found {0} unique words: ".format(len(word_index)))

data = pad_sequences(sequences, maxlen=maxLen)
print("Data Shape :", data.shape)

np.random.seed(103)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

text_train = data[:training_samples]
y_train = data[:training_samples]
text_test = data[training_samples:]
y_test = data[training_samples:]

#Build & Train RNN
model = Sequential()
model.add(Embedding(max_feature, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model.fit(text_train, y_train, epochs=10, batch_size=60, validation_split=0.2)

pred = model.predict(text_test)
pred = np.argmax(pred, axis=1)

acc = model.evaluate(text_test, y_test)
print("Test loss is {0:.2f} accuracy is {1:.2f} ".format(acc[0], acc[1]))
print(confusion_matrix(pred, y_test))


