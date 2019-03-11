# pip install Box2D
import gym

env = gym.make('LunarLander-v2')


ACTIONS = env.action_space.n


env.reset()