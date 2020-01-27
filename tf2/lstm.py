import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# https://learning.oreilly.com/videos/implementing-deep-learning/9781789950496/9781789950496-video5_2


ret = keras.datasets.reuters

(X_train, y_train), (X_test, y_test) = ret.load_data(num_words=5000)

print("X_train", X_train.shape)
print("X_test", X_test.shape)

print(np.unique(y_train))

print("Number of words: ")
print(len(np.unique(np.hstack(X_train))))

result = [len(x) for x in X_train]
sns.boxplot(y=result)
print("Mean is", np.mean(result))

print(X_train[0])

word_index = ret.get_word_index()
reverse_word_index = {y:x for x,y in word_index.items()}

def decode_review(encoded_review):
    decoded_review = []
    for word in encoded_review:
        if word in reverse_word_index:
            decoded_review.append(reverse_word_index[word])
    return decoded_review

print(decode_review(X_train[0]))


word_index = {x:(y+3) for x,y in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
reverse_word_index = {y:x for x,y in word_index.items()}

print(" ".join(decode_review(X_train[0])))
print("-------")
print(" ".join(decode_review(X_train[3])))


from tensorflow.python.keras.preprocessing import sequence

max_review_length = 500
X_train_padded = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test_padded = sequence.pad_sequences(X_test, maxlen=max_review_length)


from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_word=5000)
token_X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
token_X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')


from tensorflow.python.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN, Dropout, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam

embedding_vector_length = 512

model = Sequential()

model.add(Embedding(5000, embedding_vector_length, input_length=max_review_length))

model.add(LSTM(100, return_sequences=True))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(46, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.1)

model.summary()

model.save_weights('sentimental.h5')




val_loss = model.history.history['val_loss']
tra_loss = model.history.history['loss']

plt.plot(val_loss)
plt.plot(tra_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.legend(["Validation Loss", "Training Loss"])
plt.show()



val_acc = model.history.history['val_acc']
tra_acc = model.history.history['acc']

plt.plot(val_acc)
plt.plot(tra_acc)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.legend(["Validation Accuracy", "Training Accuracy"])
plt.show()