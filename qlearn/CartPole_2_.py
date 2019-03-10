import tensorflow as tf

def build_network(self):
    # Create placeholders
    with tf.name_scope('inputs'):
        self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
        self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
        self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")
    # Initialize parameters
    units_layer_1 = 10
    units_layer_2 = 10
    units_output_layer = self.n_y
    with tf.name_scope('parameters'):
        W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    # Forward prop
    with tf.name_scope('layer_1'):
        Z1 = tf.add(tf.matmul(W1,self.X), b1)
        A1 = tf.nn.relu(Z1)
    with tf.name_scope('layer_2'):
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
    with tf.name_scope('layer_3'):
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.softmax(Z3)
    # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
    logits = tf.transpose(Z3)
    labels = tf.transpose(self.Y)
    self.outputs_softmax = tf.nn.softmax(logits, name='A3')
with tf.name_scope('loss'):
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss
with tf.name_scope('train'):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)