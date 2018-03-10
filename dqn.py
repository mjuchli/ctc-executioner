import logging
import copy
import numpy as np
from action_space import ActionSpace
from order_side import OrderSide
from order_type import OrderType
from orderbook import Orderbook
from action_state import ActionState
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import optimizers
import random
from action_state import ActionState
import pprint
import datetime
from collections import deque
#logging.basicConfig(level=logging.DEBUG)


side = OrderSide.SELL
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -10, -12, -15]
ai = None
T = [0, 10, 20, 40, 60, 80, 100] #, 120, 240]
T_test = [0, 10, 20, 40, 60, 80, 100]# 120, 240]
I = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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

# f = getBidAskFeatures(d, 100, 0.5, 1, 10)
# f.shape
# f
# f.reshape((100,4))


def bidAskFeature(bids, asks, inventory, bestAsk, size=100):
    """Creates feature in form of two vectors representing bids and asks.

    The prices and sizes of the bids and asks are normalized by the provided
    (current) bestAsk and inventory respectively.
    """

    def normalize(d, inventory, bestAsk):
        s = pd.Series(d, name='size')
        s.index.name='price'
        s = s.reset_index()
        s.price = s.price / bestAsk
        s['size'] = s['size'] / inventory
        return np.array(s)

    def force_size(a, n=size):
        gap = (n - a.shape[0])
        if gap > 0:
            gapfill = np.zeros((gap, 2))
            a = np.vstack((a, gapfill))
            return a
        elif gap < 0:
            raise Exception('TODO: shrink')

    bids_norm = normalize(bids, inventory, bestAsk)
    asks_norm = normalize(asks, inventory, bestAsk)
    return np.array([force_size(bids_norm), force_size(asks_norm)])

def getBidAskFeatures(d, state_index, inventory, lookback, size=100):
    """
    (2*lookback, size, 2)
    """

    state = d[list(d.keys())[state_index]]
    asks = state['asks']
    bids = state['bids']
    bestAsk = min(asks.keys())
    i = 0
    while i < lookback:
        state_index = state_index - 1
        state = d[list(d.keys())[state_index]]
        asks = state['asks']
        bids = state['bids']

        if i == 0:
            features = bidAskFeature(bids, asks, inventory, bestAsk)
        else:
            features_next = bidAskFeature(bids, asks, inventory, bestAsk)
            features = np.vstack((features, features_next))
        i = i + 1
    return features

class DQNAgent:
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
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(100, input_shape=(100,20), activation='relu'))
        model.add(Dense(10, activation='relu'))
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
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def guess(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            for t in self.env.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.env.I:
                    logging.info("     i=="+str(i))
                    print("Action run " + str((t, i)))
                    action, done = self.start(t, i)
                    while not done:
                        if force_execution:
                            raise Exception("Enforced execution left " + str(i_next) + " unexecuted.")
                        t = action.getState().getT()
                        i = action.getState().getI()
                        print("Action continue " + str((t, i)))
                        action, done = self.update(action)
                        t_next = action.getState().getT()
                        i_next = action.getState().getI()
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))

            # train the agent with the experience of the episode
            print("\nREPLAY\n")
            self.replay(32)

    def start(self, t, i):
        orderbookState, orderbookIndex = self.env.getRandomOrderbookState()
        bidAskFeature = getBidAskFeatures(d, orderbookIndex, i, lookback)
        state = ActionState(t, i, {'bidask': bidAskFeature}) #np.array([[t, i]])
        a = self.act(state.toArray())
        action = self.env.createAction(a, state, orderbookIndex=orderbookIndex)
        return self.update(action)

    def update(self, action):
        state = copy.deepcopy(action.getState())

        # Run and evaluate action
        action.run(self.env.orderbook)
        i_next = self.env.determineNextInventory(action)
        t_next = self.env.determineNextTime(action.getState().getT())
        reward = action.getValueAvg(fees=False)
        bidAskFeature = getBidAskFeatures(d, action.getOrderbookIndex(), i_next, lookback)
        state_next = ActionState(t_next, i_next, {'bidask': bidAskFeature})

        # Remember the previous state, action, reward, and done
        done = action.isFilled() or state_next.getI() == 0
        self.remember(state.toArray(), action.getA(), reward, state_next.toArray(), done)

        # Update action values
        a = self.act(state_next.toArray())
        action = self.env.updateAction(action, a, state_next)
        return action, done

    def backtest(self, epochs, average=False, fixed_a=None):
        Ms = []
        #T = self.T[1:len(self.T)]
        for t in [self.env.T[-1]]:
            logging.info("\n"+"t=="+str(t))
            for i in [self.env.I[-1]]:
                logging.info("     i=="+str(i))
                actions = []

                orderbookState, orderbookIndex = self.env.getRandomOrderbookState()
                bidAskFeature = getBidAskFeatures(d, orderbookIndex, i, lookback)
                state = ActionState(t, i, {'bidask': bidAskFeature}) #np.array([[t, i]])
                a = self.guess(state.toArray())
                action = self.env.createAction(a, state, orderbookIndex=orderbookIndex)

                #print(state)
                print("t: " + str(t))
                print("i: " + str(i))
                print("Action: " + str(a))

                actions.append(a)
                action.setA(a)
                midPrice = action.getReferencePrice()

                #print("before...")
                #print(action)
                action.run(self.env.orderbook)
                #print("after...")
                #print(action)
                i_next = self.env.determineNextInventory(action)
                t_next = self.env.determineNextTime(t)
                # print("i_next: " + str(i_next))
                while i_next != 0:
                    bidAskFeature = getBidAskFeatures(d, action.getOrderbookIndex(), i_next, lookback)

                    state_next = ActionState(t_next, i_next, {'bidask': bidAskFeature})
                    a_next = self.guess(state_next.toArray())
                    # print("Q action for next state " + str(state_next) + ": " + str(a_next))
                    print("t: " + str(t_next))
                    print("i: " + str(i_next))
                    print("Action: " + str(a_next))

                    actions.append(a_next)
                    #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")

                    runtime_next = self.env.determineRuntime(t_next)
                    action.setState(state_next)
                    action.update(a_next, runtime_next)
                    action.run(self.env.orderbook)
                    #print(action)
                    i_next = self.env.determineNextInventory(action)
                    t_next = self.env.determineNextTime(t_next)

                price = action.getAvgPrice()
                # TODO: last column is for for the BUY scenario only
                if action.getOrder().getSide() == OrderSide.BUY:
                    profit = midPrice - price
                else:
                    profit = price - midPrice
                Ms.append([state, midPrice, actions, price, profit])
        if not average:
            return Ms
        return self.env.averageBacktest(Ms)


def run_profit(epochs_train=1, epochs_test=1):
    if epochs_train > 0:
        agent.train(epochs_train)
    M = agent.backtest(epochs_test, average=False)
    M = np.array(M)
    # print(M)
    return np.mean(M[0:, 4])

lookback = 5
actionSpace = ActionSpace(orderbook, side, T, I, levels=levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, levels=levels)
agent = DQNAgent(actionSpace)

from ui import UI
UI.animate(run_profit, interval=100)
