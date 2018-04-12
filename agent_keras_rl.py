from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import logging
import numpy as np
from order_side import OrderSide
from orderbook import Orderbook
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from keras import optimizers
import random
from collections import deque
import gym
#logging.basicConfig(level=logging.DEBUG)


# Load orderbook
orderbook = Orderbook()
orderbook.loadFromEvents('ob-1-small.tsv')
orderbook_test = orderbook
#orderbook.plot()

import gym_ctc_executioner
env = gym.make("ctc-executioner-v0")
env.configure(orderbook)

actions = env.levels
action_size = len(env.levels)


# Neural Net for Deep-Q learning Model
model = Sequential()
model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
model.add(Dense(env.bookSize))
model.add(Activation('relu'))
model.add(Dense(action_size))
model.add(Activation('linear'))
#model.compile(optimizers.SGD(lr=.1), "mae")
model.summary()


policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=100, visualize=True, verbose=2)
