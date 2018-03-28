import logging
import numpy as np
from order_side import OrderSide
from orderbook import Orderbook
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers
import random
from collections import deque
import gym
#logging.basicConfig(level=logging.DEBUG)

class AgentDQN:
    def __init__(self, env): #, state_size, action_size):
        # self.state_size = state_size
        self.env = env
        self.actions = env.levels
        self.action_size = len(env.levels)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.dim = 4 * self.env.lookback
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(2*self.env.bookSize, input_shape=(self.env.bookSize, self.dim), activation='relu'))
        #model.add(Dense(10, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size))
        model.compile(optimizers.SGD(lr=.1), "mse")
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        return self.guess(state)

    def guess(self, state):
        act_values = self.model.predict(state)
        action =  np.argmax(act_values[0])
        return action

    def replay(self):
        batch_size = len(self.env.T) * (len(self.env.I) - 1)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
               target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            action_index = self.actions.index(action)
            target_f[0][action_index] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            print('loss: ' + str(history.history['loss']))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            for t in self.env.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.env.I[1:]:
                    logging.info("     i=="+str(i))
                    #print("Action run " + str((t, i)))
                    state = self.env._reset(t, i)
                    action = self.act(state)
                    state_next, reward, done, _ = self.env.step(action)
                    self.remember(state, action, reward, state_next, done)
                    while not done:
                        #print("Action update")
                        state = state_next
                        action = self.act(state)
                        state_next, reward, done, _ = self.env.step(action)
                        self.remember(state, action, reward, state_next, done)

            # train the agent with the experience of the episode
            print("\nREPLAY\n")
            self.replay()


    def backtest(self, episodes=1, fixed_a=None):
        Ms = []
        t = self.env.T[-1]
        i = self.env.I[-1]
        for episode in range(int(episodes)):
            actions = []
            state = self.env._reset(t, i)
            action = self.guess(state)
            state_next, reward, done, _ = self.env.step(action)
            actions.append(action)
            midPrice = self.env.execution.getReferencePrice()
            while not done:
                action_next = self.guess(state_next)
                # print("Q action for next state " + str(state_next) + ": " + str(a_next))
                i_next = self.env.actionState.getI()
                t_next = self.env.actionState.getT()
                print("t: " + str(t_next))
                print("i: " + str(i_next))
                print("Action: " + str(action_next))
                actions.append(action_next)
                #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")
                state_next, reward, done, _ = self.env.step(action_next)
                #print(action)

            price = self.env.execution.getAvgPrice()
            if self.env.execution.getOrder().getSide() == OrderSide.BUY:
                profit = midPrice - price
            else:
                profit = price - midPrice
            Ms.append([state, midPrice, actions, price, profit])
        return Ms

    def run(self, epochs_train=1, epochs_test=10):
        if epochs_train > 0:
            agent.train(episodes=epochs_train)
        M = agent.backtest(episodes=epochs_test)
        M = np.array(M)
        return np.mean(M[0:, 4])

    def simulate(self, epochs_train=1, epochs_test=10, interval=100):
        from ui import UI
        UI.animate(lambda : self.run(epochs_train, epochs_test), interval=interval)


# Load orderbook
cols = ["ts", "seq", "size", "price", "is_bid", "is_trade", "ttype"]
import pandas as pd
events = pd.read_table('ob-1-small.tsv', sep='\t', names=cols, index_col="seq")
d = Orderbook.generateDictFromEvents(events)
orderbook = Orderbook()
orderbook.loadFromDict(d)
# clean first n states (due to lack of bids and asks)
print("#States: " + str(len(orderbook.states)))
for i in range(100):
    orderbook.states.pop(0)
    del d[list(d.keys())[0]]
orderbook_test = orderbook
#orderbook.plot()

import gym_ctc_executioner
env = gym.make("ctc-executioner-v0")
env.configure(orderbook, d)

agent = AgentDQN(env=env)
agent.simulate()
