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
from ctc_executioner.feature_type import FeatureType

def main():
    side = OrderSide.SELL
    dataset = "1"
    file_name_prefix = "cnn_"+str(dataset)+"_"+str(side)
    # Load orderbook
    orderbook = Orderbook()
    orderbook.loadFromEvents('data/events/ob-'+dataset+'-small-train.tsv')

    # import datetime
    # orderbook = Orderbook()
    # config = {
    #     'startPrice': 10000.0,
    #     # 'endPrice': 9940.0,
    #     'priceFunction': lambda p0, s, samples: p0 + 10 * np.sin(2*np.pi*10 * (s/samples)),
    #     'levels': 50,
    #     'qtyPosition': 0.1,
    #     'startTime': datetime.datetime.now(),
    #     'duration': datetime.timedelta(seconds=1000),
    #     'interval': datetime.timedelta(seconds=1)
    # }
    # orderbook.createArtificial(config)
    # orderbook.summary()
    #orderbook.plot(show_bidask=True)

    env = gym.make("ctc-executioner-v0")
    #env = gym.make("ctc-marketmaker-v0")

    #liveplot = LivePlotCallback(nb_episodes=10000, avgwindow=10)
    #liveplot.plot()

    actionRewardLog = ActionRewardLog(file_name_prefix=file_name_prefix)

    env._configure(
        orderbook=orderbook,
        callbacks=[actionRewardLog],#liveplot,
        side=side,
        featureType=FeatureType.ORDERS
    )
    print(env.observation_space.shape)
    model = deepq.models.cnn_to_mlp( convs=[(int(env.observation_space.shape[1]/2), int(env.observation_space.shape[1]/2), env.observation_space.shape[0])], hiddens=[200])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=50000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        #target_network_update_freq=1,
        print_freq=10,
        #callback=liveplot.baseline_callback
    )
    print("Saving model as "+file_name_prefix+".pkl")
    act.save("models/"+file_name_prefix+".pkl")


if __name__ == '__main__':
    main()
