import logging
import numpy as np
from action_space import ActionSpace
from order_side import OrderSide
from orderbook import Orderbook
from action_state import ActionState
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import random
from experience_replay import ExperienceReplay

from action_state import ActionState
import pprint
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set(color_codes=True)

side = OrderSide.SELL
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -10, -12, -15]
ai = None

#trainBook = 'query_result_train_15m.tsv'
#testBook = 'query_result_train_15m.tsv'
orderbook = Orderbook(extraFeatures=True)
orderbook.loadFromBitfinexFile('orderbook_bitfinex_btcusd_view.tsv')
orderbook_test = Orderbook(extraFeatures=True)
orderbook_test.loadFromBitfinexFile('orderbook_bitfinex_btcusd_view.tsv')
orderbook.createFeatures()
orderbook_test.createFeatures()
# orderbook.plot()

T = [10, 20, 40, 60, 80, 100] #, 120, 240]
T_test = [0, 10, 20, 40, 60, 80, 100]# 120, 240]
I = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def baseline_model(num_actions,hidden_size):
    #seting up the model with keras
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(optimizers.SGD(lr=.1), "mse")
    return model

# parameters
epsilon = .1  # exploration
num_actions = len(levels)  # [move_left, stay, move_right]
max_memory = 500 # Maximum number of experiences we are storing
hidden_size = 100 # Size of the hidden layers
batch_size = 1 # Number of experiences we use for training per batch

#Define model
model = baseline_model(num_actions,hidden_size)
model.summary()

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)


class ActionSpaceDQN(ActionSpace):

    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            loss = 0.
            for t in self.T:

                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    logging.info("Action run " + str((t, i)))
                    (t_next, i_next, loss) = self.update(t, i, loss, force_execution)
                    while i_next != 0:
                        if force_execution:
                            raise Exception("Enforced execution left " + str(i_next) + " unexecuted.")
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))
                        (t_next, i_next, loss) = self.update(t_next, i_next, loss, force_execution)

    def update(self, t, i, loss, force_execution=False):
        #The learner is acting on the last observed game screen
        #input_t is a vector containing representing the game screen
        state = np.array([[t, i]])
        #state = np.array([t, i]).reshape(2,1)

        #a = random.choice(levels)
        if np.random.rand() <= epsilon:
            #Eat something random from the menu
            a = random.choice(levels)
        else:
            #Choose yourself
            #q contains the expected rewards for the actions
            q = model.predict(state)
            #We pick the action with the highest expected reward
            a = np.argmax(q[0])

        action = self.createAction(a, t, i, force_execution=force_execution)
        action.run(self.orderbook)
        i_next = self.determineNextInventory(action)
        t_next = self.determineNextTime(t)
        reward = action.getValueAvg(fees=False)
        # reward = action.getValueExecuted()
        # reward = action.getTestReward()
        state_next = ActionState(action.getState().getT(), action.getState().getI(), action.getState().getMarket())
        state_next.setT(t_next)
        state_next.setI(i_next)

        # store experience
        exp_replay.remember([np.array([[t_next, i_next]]), a, reward, state], t==0)

        # Load batch of experiences
        inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

        # train model on experiences
        # print("input, targets")
        # print(inputs)
        # print(targets)
        batch_loss = model.train_on_batch(inputs, targets)

        loss += batch_loss

        return (t_next, i_next, loss)

    def backtest(self, q=None, episodes=10, average=False, fixed_a=None):
        Ms = []
        #T = self.T[1:len(self.T)]
        for t in [self.T[-1]]:
            logging.info("\n"+"t=="+str(t))
            for i in [self.I[-1]]:
                logging.info("     i=="+str(i))
                actions = []
                state = ActionState(t, i, {})
                #print(state)
                q = model.predict(np.array([[t, i]]))
                #We pick the action with the highest expected reward
                a = np.argmax(q[0])
                print(a)

                actions.append(a)
                action = self.createAction(a, t, i, force_execution=False)
                midPrice = action.getReferencePrice()

                #print("before...")
                #print(action)
                action.run(self.orderbook)
                #print("after...")
                #print(action)
                i_next = self.determineNextInventory(action)
                t_next = self.determineNextTime(t)
                # print("i_next: " + str(i_next))
                while i_next != 0:
                    state_next = ActionState(t_next, i_next, {})
                    q = model.predict(np.array([[t_next, t_next]]))
                    a_next = np.argmax(q[0])
                    # print("Q action for next state " + str(state_next) + ": " + str(a_next))

                    actions.append(a_next)
                    #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")

                    runtime_next = self.determineRuntime(t_next)
                    action.setState(state_next)
                    action.update(a_next, runtime_next)
                    action.run(self.orderbook)
                    #print(action)
                    i_next = self.determineNextInventory(action)
                    t_next = self.determineNextTime(t_next)

                price = action.getAvgPrice()
                # TODO: last column is for for the BUY scenario only
                if action.getOrder().getSide() == OrderSide.BUY:
                    profit = midPrice - price
                else:
                    profit = price - midPrice
                Ms.append([state, midPrice, actions, price, profit])
        if not average:
            return Ms
        return self.averageBacktest(Ms)





def train(episodes=100):
    for episode in range(episodes):
        # pp.pprint("Episode " + str(episode))
        actionSpace.train(episodes=1, force_execution=False)


def test(episodes=100, average=True):
    M = actionSpace_test.backtest(episodes, average=average)
    return M


def animate(f, interval=5000, axis=[0, 100, -50, 50], frames=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.axis(axis)
    ax1.autoscale(True)
    xs = []
    ys = []

    def do_animate(i, f, ax1, xs, ys):
        y = f()
        if len(xs) == 0:
            xs.append(0)
        else:
            xs.append(xs[-1]+1)
        ys.append(y)
        ax1.clear()
        ax1.plot(xs, ys)

    ani = animation.FuncAnimation(
        fig,
        lambda i: do_animate(i, f, ax1, xs, ys),
        interval=interval,
        frames=frames
    )
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())
    plt.show()


def run_profit(epochs_train=10, epochs_test=5):
    if epochs_train > 0:
        train(epochs_train)
    M = test(epochs_test, average=False)
    M = np.array(M)
    # print(M)
    return np.mean(M[0:, 4])


actionSpace = ActionSpaceDQN(orderbook, side, T, I, ai, levels)
actionSpace_test = ActionSpaceDQN(orderbook_test, side, T_test, I, ai, levels)

animate(run_profit, interval=100)
