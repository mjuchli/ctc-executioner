import logging
import numpy as np
from ctc_executioner.order_side import OrderSide
from ctc_executioner.orderbook import Orderbook
from ctc_executioner.agent_utils.ui import UI
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers
import random
from collections import deque
import gymnasium as gym

# logging.basicConfig(level=logging.DEBUG)


class AgentDQN:
    def __init__(self, env):  # , state_size, action_size):
        # self.state_size = state_size
        self.env = env
        # Get unwrapped environment for custom attributes
        self.unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.actions = self.unwrapped_env.levels
        self.action_size = len(self.unwrapped_env.levels)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 32  # len(self.env.T) * (len(self.env.I) - 1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Flatten(input_shape=self.env.observation_space.shape))
        model.add(Dense(self.unwrapped_env.bookSize))
        model.add(Dense(self.action_size))
        model.compile(optimizers.SGD(learning_rate=self.learning_rate), "mae")
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        return self.guess(state)

    def guess(self, state):
        # Ensure state has batch dimension
        if len(state.shape) == len(self.env.observation_space.shape):
            state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        # print(act_values)
        action = np.argmax(act_values[0])
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # print("reward: " + str(reward))
            if not done:
                # print("not done")
                # rewards_next = self.model.predict(next_state)
                # print("state_next: " + str(next_state))
                # print('rewards_next ' + str(rewards_next))
                # print('reward_next ' + str(np.amax(self.model.predict(next_state)[0])))
                target = reward + self.gamma * np.amax(
                    self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                )

            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            # action_index = self.actions[action]
            target_f[0][action] = target
            history = self.model.fit(
                np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0
            )
            print("loss: " + str(history.history["loss"][0]))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            for t in self.unwrapped_env.T:
                logging.info("\n" + "t==" + str(t))
                for i in self.unwrapped_env.I[1:]:
                    logging.info("     i==" + str(i))
                    # print("Action run " + str((t, i)))
                    # Reset using unwrapped env's internal method
                    state = self.unwrapped_env._reset(t, i)
                    # Reset the wrapped env to sync state (returns observation, info)
                    obs, _ = self.env.reset()
                    action = self.act(state)
                    state_next, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    self.remember(state, action, reward, state_next, done)
                    while not done:
                        # print("Action update")
                        state = state_next
                        action = self.act(state)
                        state_next, reward, terminated, truncated, _ = self.env.step(
                            action
                        )
                        done = terminated or truncated
                        self.remember(state, action, reward, state_next, done)

            # train the agent with the experience of the episode
            print("\nREPLAY\n")
            self.replay()

    def backtest(self, episodes=1, fixed_a=None):
        Ms = []
        t = self.unwrapped_env.T[-1]
        i = self.unwrapped_env.I[-1]
        for episode in range(int(episodes)):
            actions = []
            state = self.unwrapped_env._reset(t, i)
            action = self.guess(state)
            state_next, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            actions.append(action)
            midPrice = self.unwrapped_env.execution.getReferencePrice()
            while not done:
                action_next = self.guess(state_next)
                # print("Q action for next state " + str(state_next) + ": " + str(a_next))
                i_next = self.unwrapped_env.actionState.getI()
                t_next = self.unwrapped_env.actionState.getT()
                print("t: " + str(t_next))
                print("i: " + str(i_next))
                print("Action: " + str(action_next))
                actions.append(action_next)
                # print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")
                state_next, reward, terminated, truncated, _ = self.env.step(
                    action_next
                )
                done = terminated or truncated
                # print(action)

            price = self.unwrapped_env.execution.getAvgPrice()
            if self.unwrapped_env.execution.getOrder().getSide() == OrderSide.BUY:
                profit = midPrice - price
            else:
                profit = price - midPrice
            Ms.append([state, midPrice, actions, price, profit])
        return Ms

    def run(self, epochs_train=1, epochs_test=10):
        if epochs_train > 0:
            self.train(episodes=epochs_train)
        M = self.backtest(episodes=epochs_test)
        # Extract only the profit values (index 4) from each episode result
        profits = [episode[4] for episode in M]
        return np.mean(profits)

    def simulate(self, epochs_train=1, epochs_test=10, interval=100):
        UI.animate(lambda: self.run(epochs_train, epochs_test), interval=interval)


# Load orderbook
import datetime

# Try to load from file, otherwise create artificial data
orderbook = Orderbook()
try:
    orderbook.loadFromEvents("data/events/ob-train.tsv")
    print("Loaded orderbook from data/events/ob-train.tsv")
except (FileNotFoundError, IOError):
    print("Data file not found, creating artificial orderbook...")
    config = {
        "startPrice": 10000.0,
        "priceFunction": lambda p0, s, samples: p0
        + 10 * np.sin(2 * np.pi * 10 * (s / samples)),
        "levels": 50,
        "qtyPosition": 0.1,
        "startTime": datetime.datetime.now(),
        "duration": datetime.timedelta(seconds=1000),
        "interval": datetime.timedelta(seconds=1),
    }
    orderbook.createArtificial(config)
    orderbook.summary()

orderbook_test = orderbook
# orderbook.plot()

import gym_ctc_executioner

env = gym.make("ctc-executioner-v0")
# Unwrap environment to access custom methods
unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
unwrapped_env.setOrderbook(orderbook)

agent = AgentDQN(env=env)
agent.simulate(1)
# agent.train(10)
