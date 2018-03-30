import gym
from orderbook import Orderbook
from baselines import deepq
import gym_ctc_executioner


def main():
    # Load orderbook
    orderbook = Orderbook()
    orderbook.loadFromEvents('ob-1-small.tsv')


    env = gym.make("ctc-executioner-v0")
    env.configure(orderbook)
    model = deepq.models.cnn_to_mlp( convs=[(1, 10, 20)], hiddens=[200])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=100000,
        buffer_size=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        target_network_update_freq=1,
        print_freq=10,
    )
    print("Saving model as ctc-executioner-v0.pkl")
    act.save("ctc-executioner-v0.pkl")


if __name__ == '__main__':
    main()
