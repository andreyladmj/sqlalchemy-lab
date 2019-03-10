import tensorflow as tf
import numpy as np


x = tf.Variable(4)
x = tf.assign(x, 2)

holder1 = tf.placeholder(tf.float32)
holder_sum = holder1 + 2

sess = tf.Session()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(x)


    print(sess.run(tf.multinomial(tf.log([[0.1, 0.1, 0.1]]), num_samples=1)))

    print(sess.run(tf.multinomial([[np.log(0.5), np.log(0.2), np.log(0.3)]], num_samples=5)))
    test = sess.run(tf.multinomial([[np.log(0.5), np.log(0.2), np.log(0.3)]], num_samples=500000))
    # first action probability is 50%, second - 20%, 3 - 30%

test = np.array(test[0])
np.sum(test == 0) / test.shape[0]
np.sum(test == 1) / test.shape[0]
np.sum(test == 2) / test.shape[0]




# https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient.py

X = tf.placeholder(tf.float32, shape=(None, 5), name="X")
Y = tf.placeholder(tf.float32, shape=(None, 3), name="Y")
discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")
W1 = tf.get_variable("W1", [5, 3], initializer = tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [3], initializer = tf.contrib.layers.xavier_initializer(seed=1))
Z1 = tf.add(tf.matmul(X, W1), b1)

logits = tf.transpose(Z1)
labels = tf.transpose(Y)
outputs_softmax = tf.nn.softmax(logits, name='A3')
neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_norm)  # reward guided loss
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

sess.run(tf.global_variables_initializer())

outputs_softmax_ = sess.run([outputs_softmax], feed_dict={X: np.array([[0,0,1,0,0]])})
outputs_softmax_ = outputs_softmax_[0].reshape((1,3))

outputs_softmax_.shape

X_, Y_, discounted_episode_rewards_norm_, logits_, labels_, outputs_softmax_2, neg_log_prob_, loss_, train_op_ = sess.run([
    X, Y, discounted_episode_rewards_norm, logits, labels, outputs_softmax, neg_log_prob, loss, train_op
], feed_dict={X: np.array([[1,1,1,1,1]]), Y:outputs_softmax_, discounted_episode_rewards_norm:np.array([1,0,0])})

#
# X_, Y_, discounted_episode_rewards_norm_, logits_, labels_, outputs_softmax_2, neg_log_prob_, loss_, train_op_ = sess.run([
#     X, Y, discounted_episode_rewards_norm, logits, labels, outputs_softmax, neg_log_prob, loss, train_op
# ], feed_dict={X: np.array([[1,1,1,1,1]]), Y:outputs_softmax_, discounted_episode_rewards_norm:np.array([1,0,0])})


print('X_', X_)
print('Y_', Y_)
print('discounted_episode_rewards_norm_', discounted_episode_rewards_norm_)
print('logits_', logits_)
print('labels_', labels_)
print('outputs_softmax_', outputs_softmax_)
print('neg_log_prob_', neg_log_prob_)
print('loss_', loss_)
print('train_op_', train_op_)

