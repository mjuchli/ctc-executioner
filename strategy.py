import logging
import numpy as np
from action_space import ActionSpace
from qlearn import QLearn
from order_side import OrderSide
from orderbook import Orderbook
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def train(episodes=100, f='query_result_small.tsv'):
    if not orderbook.getStates():
        orderbook.loadFromFile(f)
    actionSpace = ActionSpace(orderbook, side, V, T, I, H, ai, levels)

    for episode in range(episodes):
        # pp.pprint("Episode " + str(episode))
        actionSpace.train(episodes=1, force_execution=False)
        np.save('q.npy', actionSpace.ai.q)
        # pp.pprint(actionSpace.ai.q)
    return actionSpace.ai.q


def test(episodes=100, f='query_result_small.tsv'):
    if not orderbook_test.getStates():
        orderbook_test.loadFromFile(f)

    actionSpace_test = ActionSpace(orderbook_test, side, V, T_test, I, H, ai, levels)

    q = np.load('q.npy').item()
    # M <- [t, i, Price, A, Paid, Diff]
    M = actionSpace_test.backtest(q, episodes, average=True)
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
    q = train(10)
    M = test(10)
    M = np.array(M)
    return np.mean(M[0:, 4])

def run_q_reward():
    q = train(1)
    return np.mean(list(q.values()))

pp = pprint.PrettyPrinter(indent=4)
#logging.basicConfig(level=logging.INFO)

side = OrderSide.BUY
V = 1.0
# T = [4, 3, 2, 1, 0]
T = [0, 10, 30, 60]
T_test = [60]
# I = [1.0, 2.0, 3.0, 4.0]
I = [0.1, 0.3, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
H = max(I)
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -7, -10, -15, -20, -25, -30, -50]
ai = QLearn(actions=levels, epsilon=0.2, alpha=0.5, gamma=0.8)
orderbook = Orderbook()
orderbook_test = Orderbook()



#animate(run_profit, interval=1000)
animate(run_q_reward, interval=1000)
