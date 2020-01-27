import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

observations = 1000

xs=np.random.uniform(low=-10,high=10,size=(observations,1))
zs=np.random.uniform(low=-10,high=10,size=(observations,1))


inputs = np.column_stack((xs,zs))

# targets = f(x,z) = 2*x - 3*z + 5 + noise
noise = np.random.uniform(-1,1,(observations,1))
targets = 2*xs - 3*zs + 5 + noise


def plot(targets):
    targets = targets.reshape(observations,)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, zs, targets)
    ax.set_xlabel('xs')
    ax.set_ylabel('zs')
    ax.set_zlabel('Targets')
    ax.view_init(azim=100)
    plt.show()

plot(targets)


for i in (100,):
    print(i)

np.savez("TF_intor", inputs=inputs, targets=targets)

training_data = np.load('TF_intro.npz')
imput_size = 2
output_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        output_size,
        kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
        bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    )
])

# model.compile(optimizer='sgd', loss='mean_squared_error')

custom_initializer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer='sgd', loss='mean_squared_error')


model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=0)

# Extract the weights and bias
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]

# Predict the outputs (make predictions)
model.predict_on_batch(training_data['inputs'])
model.predict_on_batch(training_data['inputs']).round(1)


# Plot the data
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))


x = np.arange(10)
np.save(outfile, x)

#https://github.com/python/cpython/blob/master/Modules/_pickle.c
# PCA https://books.google.com.ua/books?id=BVnHDwAAQBAJ&pg=PA376&lpg=PA376&dq=tensorflow+2+pca&source=bl&ots=KZyc8mXx2-&sig=ACfU3U1AIDZVxWfwNGC92BbBKxc8dU4StA&hl=en&sa=X&ved=2ahUKEwjjr8uC44znAhXPpIsKHTn8A1oQ6AEwCHoECAoQAQ#v=onepage&q=tensorflow%202%20pca&f=false