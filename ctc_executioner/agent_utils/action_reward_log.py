import numpy as np
import matplotlib.pyplot as plt
from rl.callbacks import Callback

class ActionRewardLog(Callback):
    def __init__(self, file_name_prefix, nb_episodes=10000, avgwindow=20):
        self.rewards = np.zeros(nb_episodes) - 1000.0
        self.X = np.arange(1, nb_episodes+1)
        self.avgrewards = np.zeros(nb_episodes) - 1000.0
        self.avgwindow = avgwindow
        self.rewardbuf = []
        self.episode = 0
        self.nb_episodes = nb_episodes
        self.file_actions = file_name_prefix + "_actions.py"
        self.file_rewards = file_name_prefix + "_rewards.py"
        self.file_rewards_mean = file_name_prefix + "_rewards_mean.py"

    def on_episode_end(self, episode, logs):
        if self.episode >= self.nb_episodes:
            return
        rw = logs['episode_reward']
        actions = logs['episode_actions']
        self.rewardbuf.append(rw)
        if len(self.rewardbuf) > self.avgwindow:
            del self.rewardbuf[0]
        self.rewards[self.episode] = rw
        rw_avg = np.mean(self.rewardbuf)
        self.avgrewards[self.episode] = rw_avg
        self.episode += 1
        self.write(self.file_actions, actions)
        self.write(self.file_rewards, rw)
        self.write(self.file_rewards_mean, rw_avg)

    def write(self, file, value):
        f = open(file, "a")
        f.write(str(value) + ',\n')
