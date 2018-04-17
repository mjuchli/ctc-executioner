import matplotlib.pyplot as plt
import numpy as np
from rl.callbacks import Callback
from order_side import OrderSide

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
            state = self.env.orderbook.getState(index)
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
