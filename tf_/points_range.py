import tensorflow as tf


tf.reset_default_graph()

n_input = 4
n_output = 2
n_hidden_1 = 12
learning_rate = 0.001


x = tf.placeholder(tf.float32, [None, n_input], name='x')
y = tf.placeholder(tf.float32, [None, n_output], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='h1'),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_output]), name='h_out')
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'out': tf.Variable(tf.random_normal([n_output]), name='b_out')
}


def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'], name='logits')
    return out_layer

logits = multilayer_perceptron(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
tf.summary.scalar('softmax_cross_entropy_with_logits_v2', cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_pred = tf.argmax(y, 1)

def check_accuracy(k=3):
    correct_prediction = tf.nn.in_top_k(predictions=logits, targets=correct_pred, k=k)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    s, c = 0, 0
    for batch_xs, batch_ys in zip(valid_train_X, valid_train_Y):
        s += accuracy.eval({x: batch_xs.toarray(), y: batch_ys.toarray(), keep_prob: 1})
        c += 1

    train_acc = s / c

    s, c = 0, 0
    for batch_xs, batch_ys in zip(test_X, test_Y):
        s += accuracy.eval({x: batch_xs.toarray(), y: batch_ys.toarray(), keep_prob: 1})
        c += 1

    test_acc = s / c

    return (train_acc, test_acc)

with tf.Session() as sess:
    variables = []

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.

        total_batch = 0
        for batch_xs, batch_ys in zip(train_X, train_Y):
            total_batch += 1

            _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_xs.toarray()), y: np.array(batch_ys.toarray()), keep_prob: 0.5})
            avg_cost += c

        avg_cost = avg_cost / total_batch

        log("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
        save_path = saver.save(sess, model_name)
        log('Model saved', save_path)

        acc = {}
        train_1_acc, test_1_acc = check_accuracy(1)
        train_3_acc, test_3_acc = check_accuracy(3)
        train_5_acc, test_5_acc = check_accuracy(5)
        acc['train_1_acc'] = train_1_acc
        acc['test_1_acc'] = test_1_acc
        acc['train_3_acc'] = train_3_acc
        acc['test_3_acc'] = test_3_acc
        acc['train_5_acc'] = train_5_acc
        acc['test_5_acc'] = test_5_acc
        acc['cost_training'] = avg_cost
        acc['epoch'] = epoch
        log(acc)

    log("Optimization Finished!")

