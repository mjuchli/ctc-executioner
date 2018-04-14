from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback

import logging
import numpy as np
from order_side import OrderSide
from orderbook import Orderbook
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD

from keras import optimizers
import random
from collections import deque
import gym
import matplotlib.pyplot as plt

#logging.basicConfig(level=logging.DEBUG)

class EpsDecayCallback(Callback):
    def __init__(self, eps_poilcy, decay_rate=0.95):
        self.eps_poilcy = eps_poilcy
        self.decay_rate = decay_rate
    def on_episode_begin(self, episode, logs={}):
        self.eps_poilcy.eps *= self.decay_rate
        print('eps = %s' % self.eps_poilcy.eps)

class LivePlotCallback(Callback):
    def __init__(self, nb_episodes=4000, avgwindow=20):
        self.rewards = np.zeros(nb_episodes) - 1000.0
        self.X = np.arange(1, nb_episodes+1)
        self.avgrewards = np.zeros(nb_episodes) - 1000.0
        self.avgwindow = avgwindow
        self.rewardbuf = []
        self.episode = 0
        self.nb_episodes = nb_episodes
        plt.ion()
        self.fig = plt.figure()
        self.grphinst = plt.plot(self.X, self.rewards, color='b')[0]
        self.grphavg  = plt.plot(self.X, self.avgrewards, color='r')[0]
        plt.ylim([-450.0, 350.0])
        plt.xlabel('Episodes')
        plt.legend([self.grphinst, self.grphavg], ['Episode rewards', '20-episode-average-rewards'])
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='k', linestyle='--')

    def __del__(self):
        self.fig.savefig('monitor/plot.png')

    def on_episode_end(self, episode, logs):
        if self.episode >= self.nb_episodes:
            return
        rw = logs['episode_reward']
        self.rewardbuf.append(rw)
        if len(self.rewardbuf) > self.avgwindow:
            del self.rewardbuf[0]
        self.rewards[self.episode] = rw
        self.avgrewards[self.episode] = np.mean(self.rewardbuf)
        self.plot()
        self.episode += 1
    def plot(self):
        self.grphinst.set_ydata(self.rewards)
        self.grphavg.set_ydata(self.avgrewards)
        plt.draw()
        plt.pause(0.01)

class ActionPlotCallback(Callback):
    def __init__(self, nb_episodes=10):
        self.nb_episodes = nb_episodes
        self.episodes = {}
        self.episode = {}
        self.step = None
        self.plt = None

    def on_episode_begin(self, episode, logs):
        self.episode = {'episode': episode, 'steps': {}}

    def on_episode_end(self, episode, logs):
        if episode == 0:
             self.plt = self.env.orderbook.plot(show_bidask=True, max_level=0, show=False)
        self.plot(self.episode)
        if episode == (self.nb_episodes - 1):
            self.plt.show()
        self.episodes[episode] = self.episode

    def on_step_begin(self, step, logs):
        self.step = {}

    def on_step_end(self, step, logs):
        self.step['reward'] = logs['reward']
        self.episode['steps'][step] = self.step

    def on_action_begin(self, action, logs):
        self.step['action'] = action
        self.step['index'] = self.env.orderbookIndex
        self.step['t'] = self.env.actionState.getT()
        self.step['i'] = self.env.actionState.getI()

    def plot(self, episode):
        indices, times, actions, prices, order_prices, runtimes, inventories, rewards = [], [], [], [], [], [], [], []
        for key, value in episode['steps'].items():
            index = value['index']
            indices.append(index)
            runtimes.append(value['t'])
            inventories.append(value['i'])
            rewards.append(value['reward'])
            actions.append(value['action'])
            state = orderbook.getState(index)
            prices.append(state.getBidAskMid())
            times.append(state.getTimestamp())
            action_delta = 0.1*self.env.levels[value['action']]
            if self.env.side == OrderSide.BUY:
                order_prices.append(state.getBidAskMid() + action_delta)
            else:
                order_prices.append(state.getBidAskMid() - action_delta)

        # order placement
        #plt.scatter(times, prices, s=60)
        self.plt.scatter(times, order_prices, s=20)

        for i, time in enumerate(times):
            if i == 0 or i == len(times)-1:
                style = 'k-'
            else:
                style = 'k--'
            # line at order placement
            self.plt.plot([time, time], [prices[i]-0.005*prices[i], prices[i]+0.005*prices[i]], style, lw=1)

        for i, action in enumerate(actions):
            # action, resulted reward
            txt = 'a='+str(self.env.levels[action]) + '\nr=' + str(round(rewards[i], 2))
            self.plt.annotate(txt, (times[i],prices[i]))
            # runtime, inventory
            txt = 't=' + str(runtimes[i]) + '\ni='+ str(round(inventories[i], 2))
            self.plt.annotate(txt, (times[i], prices[i]-0.005*prices[i]))


# Load orderbook
orderbook = Orderbook()
orderbook.loadFromEvents('ob-1.tsv')
orderbook_test = orderbook
orderbook.summary()
#orderbook.plot()

import gym_ctc_executioner
env = gym.make("ctc-executioner-v0")
env.configure(orderbook)

actions = env.levels
action_size = len(env.levels)

def createModel():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Flatten(input_shape=(1,)+env.observation_space.shape))
    model.add(Dense(4*env.bookSize*env.lookback))
    model.add(Activation('relu'))
    model.add(Dense(action_size))
    #model.add(Activation('linear'))
    #model.compile(optimizers.SGD(lr=.1), "mae")
    model.summary()
    return model

def loadModel(name):
    # load json and create model
    from keras.models import model_from_json
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(name + '.h5')
    print('Loaded model "' + name + '" from disk')
    return model

def saveModel(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + '.h5')
    print('Saved model "' + name + '" to disk')


model = loadModel(name='model-sell-lb20-ep100000')
#model = createModel()
nrTrain = 1000
nrTest = 10

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=5000, window_length=1)
# nb_steps_warmup: the default value for that in the DQN OpenAI baselines implementation is 1000
dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
#cbs = []
#cbs = [LivePlotCallback(nb_episodes=20000, avgwindow=20)]
#dqn.fit(env, nb_steps=nrTrain, visualize=True, verbose=2, callbacks=cbs)

cbs = [ActionPlotCallback(nb_episodes=nrTest)]
dqn.test(env, nb_episodes=nrTest, visualize=True, verbose=2, callbacks=cbs)
