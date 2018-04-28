import numpy as np
import matplotlib.pyplot as plt
from rl.callbacks import Callback

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
