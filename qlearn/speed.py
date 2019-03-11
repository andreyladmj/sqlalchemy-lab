import numpy as np
import tensorflow as tf
a = 16
v0 = 20
s = 0

def get_speed(time=1):
    return v0 + a * time



# for i in range(20):
#     s += get_speed()
#     print(i, a, s)
#     a -= 4

def get_state():
    return [a, s]

def get_reward():
    if abs(s) > 100: return -3
    if abs(s) > 50: return -2
    if abs(s) > 7: return -1
    return 1

def get_actions():
    return [1, 0]


def next_state(action):
    global a
    if action == 1: a -= 1
    if action == 2: a -= 2
    if action == 3: a += 1
    if action == 4: a += 2


X = tf.placeholder(tf.float32, shape=(None, 5), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")
W1 = tf.get_variable("W1", [5, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
Z1 = tf.add(tf.matmul(X, W1), b1)

logits = tf.transpose(Z1)
labels = tf.transpose(Y)
outputs_softmax = tf.nn.softmax(logits, name='A3')
# neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
neg_log_prob = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_norm)  # reward guided loss
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.close()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(10):
        outputs_softmax_ = sess.run([outputs_softmax], feed_dict={X: np.array([get_state()])})
        outputs_softmax_ = outputs_softmax_[0].reshape((1,3))

        outputs_softmax_.shape

        X_, Y_, discounted_episode_rewards_norm_, logits_, labels_, outputs_softmax_2, neg_log_prob_, loss_, train_op_ = sess.run([
            X, Y, discounted_episode_rewards_norm, logits, labels, outputs_softmax, neg_log_prob, loss, train_op
        ], feed_dict={X: np.array([[1,1,1,1,1]]), Y:outputs_softmax_, discounted_episode_rewards_norm:np.array([1,0,0])})
