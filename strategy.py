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


def test(episodes=100, average=True, fixed_a=None):
    if not orderbook_test.getStates():
        orderbook_test.loadFromFile(testBook)

    q = np.load('q.npy').item()
    # M <- [t, i, Price, A, Paid, Diff]
    M = actionSpace_test.backtest(q, episodes, average=average, fixed_a=fixed_a)
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


def run_profit(epochs_train=0, epochs_test=5, fixed_a=None):
    if epochs_train > 0:
        q = train(epochs_train)
    M = test(epochs_test, average=False, fixed_a=fixed_a)
    M = np.array(M)
    print(M)
    return np.mean(M[0:, 4])


def calculate_profits(epochs, fixed_a=None):
    profits = []
    for i in range(epochs):
        M = test(1, average=False, fixed_a=fixed_a)
        M = np.array(M)
        #print(M)
        profits.append(np.sum(M[0:, 4]))
    return profits


def hist_profit(episodes, fixed_a=None):
    x = calculate_profits(episodes, fixed_a=fixed_a)
    sns.distplot(x)
    plt.show()


def run_q_reward():
    q = train(1)
    reward = np.mean(list(q.values()))
    print("Cummultive reward: " + str(reward))
    return reward


#logging.basicConfig(level=logging.DEBUG)

side = OrderSide.BUY
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -10, -12, -15]
ai = QLearn(actions=levels, epsilon=0.4, alpha=0.3, gamma=0.8)

#trainBook = 'query_result_train_15m.tsv'
#testBook = 'query_result_train_15m.tsv'
orderbook = Orderbook(extraFeatures=True)
orderbook.loadFromBitfinexFile('orderbook_bitfinex_btcusd_view.tsv')
orderbook_test = Orderbook(extraFeatures=True)
orderbook_test.loadFromBitfinexFile('orderbook_bitfinex_btcusd_view.tsv')

T = [10, 20, 40, 60, 80, 100] #, 120, 240]
T_test = [0, 10, 20, 40, 60, 80, 100]# 120, 240]

I = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
actionSpace = ActionSpace(orderbook, side, T, I, ai, levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, ai, levels)


def evaluateReturns(levels=range(-50, 51), crossval=10, force_execution=True):
    t = T[-1]
    i = I[-1]
    ys = []
    ys2 = []
    for level in levels:
        profit = []
        profit2 = []
        a = level
        for _ in range(crossval):
            action = actionSpace.createAction(a, t, i, force_execution=force_execution)
            refBefore = action.getReferencePrice()
            action.run(actionSpace.orderbook)
            refAfter = action.getOrderbookState().getTradePrice()
            paid = action.getAvgPrice()
            if paid == 0.0:
                assert force_execution == False
                continue
            elif action.getOrder().getSide() == OrderSide.BUY:
                profit.append(refBefore - paid)
                profit2.append(refAfter - paid)
            else:
                profit.append(paid - refBefore)
                profit2.append(paid - refAfter)

        ys.append(profit)
        ys2.append(profit2)
    x = levels
    return (x, ys, ys2)


def priceReturnCurve(levels=range(-50, 51), crossval=10, force_execution=True):
    (x, ys, ys2) = evaluateReturns(levels, crossval, force_execution)
    y = [np.mean(reject_outliers(x)) for x in ys]
    y2 = [np.mean(reject_outliers(x)) for x in ys2]
    plt.plot(x, y, 'r-')
    plt.plot(x, y2, 'g-')
    plt.show()

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


T = [0, 240] #, 120, 240
I = [10.0]
actionSpace = ActionSpace(orderbook, OrderSide.BUY, T, I)
#priceReturnCurve(crossval=10, force_execution=True)
(x, ys, ys2) = evaluateReturns(crossval=20)
for y in ys:
    print(y)
    # print(len(y))
    yr = reject_outliers(np.array(y), m=1.5)
    print(set(y) - set(yr.tolist()))
    print(len(y)-len(yr))



#train(1)
#priceReturnCurve(crossval=1)
#animate(run_profit, interval=100)
#animate(run_q_reward, interval=1000)
