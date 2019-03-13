import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

sess = tf.Session()

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
hidden_size = 32
output_size = env.action_space.n


from tensorflow.python.ops import random_ops
def _initializer(shape, dtype=tf.float32, partition_info=None):
    return random_ops.random_normal(shape)


input = tf.placeholder(tf.float32, shape=[None, input_size])

hidden1 = tf.contrib.layers.fully_connected(
    inputs=input,
    num_outputs=hidden_size,
    activation_fn=tf.nn.relu,
    weights_initializer=_initializer
)

logits = tf.contrib.layers.fully_connected(
    inputs=hidden1,
    num_outputs=output_size,
    activation_fn=None
)


# sample
random_action = tf.reshape(tf.multinomial(logits, 1), [])

log_probabilities = tf.log(tf.nn.softmax(logits))

actions = tf.placeholder(tf.int32)
advantages = tf.placeholder(tf.float32)

indices = tf.range(0, tf.shape(log_probabilities)[0]) * tf.shape(log_probabilities)[1] + actions
action_probabilities = tf.gather(tf.reshape(log_probabilities, [-1]), indices)

loss = -tf.reduce_sum(tf.multiply(action_probabilities, advantages))

optimizer = tf.train.RMSPropOptimizer(0.1)
_train = optimizer.minimize(loss)

# check random action
# sess.run(logits, feed_dict={input: [observation]})

def act(observation):
   return sess.run(random_action, feed_dict={input: [observation]})

def train_step(b_obs, b_acts, b_rews):
    batch_feed = {input: b_obs, \
                  actions: b_acts, \
                  advantages: b_rews }
    sess.run(_train, feed_dict=batch_feed)


    # checkpoint = sess.run(action_probabilities, feed_dict=batch_feed)
    sess.run(action_probabilities, feed_dict=batch_feed)
    sess.run(actions, feed_dict=batch_feed)
    sess.run(advantages, feed_dict=batch_feed)

def policy_rollout(env):
    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = act(observation)
        observation, reward, done, _ = env.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)


monitor_dir = '/tmp/cartpole_exp1'

monitor = Monitor(env, monitor_dir, force=True)

sess.run(tf.global_variables_initializer())
b_obs, b_acts, b_rews = [], [], []


# for _ in range(eparams['ep_per_batch']):

obs, acts, rews = policy_rollout(env)

print('Episode steps: {}'.format(len(obs)))

b_obs.extend(obs)
b_acts.extend(acts)

advantages_rew = process_rewards(rews)
b_rews.extend(advantages_rew)


b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)

train_step(b_obs, b_acts, b_rews)

monitor.close()




sess.close()