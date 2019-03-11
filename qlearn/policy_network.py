import numpy as np
import tensorflow as tf

observations = tf.placeholder(shape=[None, 80*80])  # pixels
actions = tf.placeholder(shape=[None])  # 0, 1, 2 for up, still, down
rewards = tf.placeholder(shape=[None])  # +1, -1 with discounts


Y = tf.layers.dense(observations, 200, activation=tf.nn.relu)
Ylogits = tf.layers.dense(Y, 3)
sample_op = tf.multinomial(logits=Ylogits, num_samples=1)

