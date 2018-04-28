import gym
from ctc_executioner.orderbook import Orderbook
from baselines import deepq
import gym_ctc_executioner

def main():
    # Load orderbook
    orderbook = Orderbook()
    orderbook.loadFromEvents('data/events/ob-1.tsv')
    env = gym.make("ctc-executioner-v0")
    env._configure(orderbook)
    model = deepq.models.cnn_to_mlp( convs=[(1, 10, 20)], hiddens=[200])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        #target_network_update_freq=1,
        print_freq=10
    )
    print("Saving model as ctc-executioner-v0.pkl")
    act.save("models/ctc-executioner-v0.pkl")


if __name__ == '__main__':
    main()
