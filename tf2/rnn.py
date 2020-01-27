import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# https://learning.oreilly.com/videos/implementing-deep-learning/9781789950496/9781789950496-video4_2
# https://learning.oreilly.com/library/view/neural-networks-and/9781492037354/ch04.html#idm139624960157312 interesting

df_date = pd.read_csv("D:\datasets\GOOGL.csv")

X_train = df_date[:200]
X_test = df_date[200:]

print(X_train.shape, X_test.shape)

X_train_high = X_train["High"]
X_test_high = X_test["High"]

X_train_high = np.array(X_train_high)
X_test_high = np.array(X_test_high)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(X_train_high.reshape(-1,1))
X_train_scaled = sc.transform(X_train_high.reshape(-1,1))


window_size = 90
def prepare_data(dataset, window_size):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
        y.append(dataset[i, 0])

    return np.array(X), np.array(y)


X_train_new, y_train_new = prepare_data(X_train_scaled, window_size)

# Get test data windows resized
df_ = df_date[len(df_date) - len(X_test) - window_size:].values
_X_test = np.array(df_[:, 1])
_X_test_scaled = sc.transform(_X_test.reshape(-1,1))
X_test_new, y_test_new = prepare_data(_X_test_scaled, window_size)
_X_test_scaled.shape



X_train_new.shape
y_train_new.shape

X_train_new = np.reshape(X_train_new, newshape=(len(X_train)-window_size, window_size, 1))
y_train_new = np.reshape(y_train_new, newshape=(len(X_train)-window_size, 1))
X_test_new = np.reshape(X_test_new, newshape=(len(X_test), window_size, 1))
y_test_new = np.reshape(y_test_new, newshape=(len(X_test), 1))

X_train_new[0].shape


model = Sequential()

model.add(SimpleRNN(50, return_sequence=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))

model.add(SimpleRNN(50, return_sequence=True))
model.add(Dropout(0.5))

model.add(SimpleRNN(50))

model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='Adam')

model_history = model.fit(X_train_new, y_train_new, epochs=20, batch_size=64)

print(model.summary())












###############################################
data = [[i for i in range(100)]]
data = np.array(data, dtype=float)

target = [[i for i in range(1,101)]]
target = np.array(target, dtype=float)

data = data.reshape((1,1,100))
target = target.reshape((1,1,100))

layers = [
    tf.keras.layers.LSTM(100, input_shape=(1, 100), return_sequences=True),
    tf.keras.layers.Dense(units=100)
]

model = tf.keras.Sequential(layers)

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.fit(data, target, nb_epoch=10000, batch_size=1, verbose=2, validation_data=(x_test, y_test))

result = model.predict(data)

print(result)
generate_numbers(1,1,5)

def generate_numbers(x1, x2, n):
    arr = [x1, x2]
    for i in range(n):
        arr.append(arr[i] + arr[i+1]*2)
    return arr

data.shape
###############################################

###############################################
#define documents
docs = ['this, is','is an']
# define class labels
labels = ['an','example']
from collections import Counter
counts = Counter()
for i,review in enumerate(docs+labels):
      counts.update(review.split())
words = sorted(counts, key=counts.get, reverse=True)
vocab_size=len(words)
word_to_int = {word: i for i, word in enumerate(words, 1)}
###############################################