import gym
from ctc_executioner.orderbook import Orderbook
from baselines import deepq
import gym_ctc_executioner
import gym_ctc_marketmaker
import numpy as np
from ctc_executioner.agent_utils.action_plot_callback import ActionPlotCallback
from ctc_executioner.agent_utils.live_plot_callback import LivePlotCallback
from ctc_executioner.agent_utils.action_reward_log import ActionRewardLog
from ctc_executioner.order_side import OrderSide

def main():
    epochs = 1000
    side = OrderSide.SELL
    dataset = "2"
    file_name_prefix = "cnn_"+str(dataset)+"_"+str(side)
    # Load orderbook
    orderbook = Orderbook()
    orderbook.loadFromEvents('data/events/ob-'+dataset+'-small-test.tsv')

    env = gym.make("ctc-executioner-v0")
    #env = gym.make("ctc-marketmaker-v0")

    #liveplot = LivePlotCallback(nb_episodes=10000, avgwindow=10)
    #liveplot.plot()
    #actionRewardLog = ActionRewardLog(file_name_prefix=file_name_prefix+'_backtest')

    env._configure(
        orderbook=orderbook,
        callbacks=[],#[liveplot],#[actionRewardLog],
        side=side
    )

    act = deepq.load("models/"+file_name_prefix+".pkl")
    rewards = []
    episode = 0
    for _ in range(epochs):
        episode += 1
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode "+str(episode)+" reward", episode_rew)
        rewards.append(episode_rew)
    print(rewards)
    print("Mean reward: " + str(np.mean(rewards)))

if __name__ == '__main__':
    main()
