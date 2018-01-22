import logging
import numpy as np
from action_space import ActionSpace
from qlearn import QLearn
from order_side import OrderSide
from orderbook import Orderbook
from action_state import ActionState
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set(color_codes=True)

def getAvgPriceDiffForInventory(M, inventory_observe):
    ms = [x for x in M if x[0][0] != 0 and x[0][1] == inventory_observe] # filter market orders (t==0)
    price_diffs = [x[4] for x in ms]
    return np.mean(price_diffs)


def getBestTimeForInventory(M, inventory_observe):
    ms = [x for x in M if x[0][1] == inventory_observe]
    # difference of the price to what was bought e.g. sold for
    price_diffs = [x[4] for x in ms]
    if side == OrderSide.BUY:
        best_price = max(price_diffs)
    else:
        best_price = min(price_diffs)
    i = price_diffs.index(best_price)
    return ms[i]


def train(episodes=100):
    if not orderbook.getStates():
        orderbook.loadFromFile(trainBook)

    for episode in range(episodes):
        # pp.pprint("Episode " + str(episode))
        actionSpace.train(episodes=1, force_execution=False)
        np.save('q.npy', actionSpace.ai.q)
        # pp.pprint(actionSpace.ai.q)
    return actionSpace.ai.q


def test(episodes=100, average=True):
    if not orderbook_test.getStates():
        orderbook_test.loadFromFile(testBook)

    q = np.load('q.npy').item()
    # M <- [t, i, Price, A, Paid, Diff]
    M = actionSpace_test.backtest(q, episodes, average=average)
    return M


def animate(f, interval=5000, axis=[0, 100, -50, 50]):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.axis(axis)
    ax1.autoscale(False)
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
        interval=interval
    )
    plt.show()


def run_profit():
    q = train(1)
    M = test(5, average=False)
    M = np.array(M)
    print(M)
    return np.mean(M[0:, 4])


def calculate_profits(epochs):
    profits = []
    for i in range(epochs):
        M = test(1, average=False)
        M = np.array(M)
        print(M)
        profits.append(np.sum(M[0:, 4]))
    return profits


def hist_profit(episodes):
    x = calculate_profits(episodes)
    sns.distplot(x)
    plt.show()


def run_q_reward():
    q = train(1)
    reward = np.mean(list(q.values()))
    print("Cummultive reward: " + str(reward))
    return reward


pp = pprint.PrettyPrinter(indent=4)
# logging.basicConfig(level=logging.DEBUG)
trainBook = 'query_result_train_15m.tsv'
testBook = 'query_result_test_15m.tsv'

side = OrderSide.BUY
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -10, -12, -15]
ai = QLearn(actions=levels, epsilon=0.4, alpha=0.3, gamma=0.8)

orderbook = Orderbook(extraFeatures=False)
orderbook_test = Orderbook(extraFeatures=False)

T = [10, 20, 40, 60, 80, 100] #, 120, 240]
T_test = [0, 20, 40, 60, 80, 100] #, 120, 240]

I = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
actionSpace = ActionSpace(orderbook, side, T, I, ai, levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, ai, levels)

train(50)
hist_profit(100)

#animate(run_profit, interval=100)
#animate(run_q_reward, interval=1000)


I = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
actionSpace = ActionSpace(orderbook, side, T, I, ai, levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, ai, levels)
train(50)
hist_profit(100)

I = [0.01, 0.02, 0.03, 0.04, 0.05]
actionSpace = ActionSpace(orderbook, side, T, I, ai, levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, ai, levels)
train(50)
hist_profit(100)
hist_profit(100)
