import logging
import numpy as np
from action_space import ActionSpace
from qlearn import QLearn
from order_side import OrderSide
from orderbook import Orderbook
import pprint

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


pp = pprint.PrettyPrinter(indent=4)

orderbook = Orderbook()
orderbook.loadFromFile('query_result_train.tsv')
orderbook_test = Orderbook()
orderbook_test.loadFromFile('query_result_test.tsv')

side = OrderSide.BUY
V = 10.0
# T = [4, 3, 2, 1, 0]
T = [0, 30, 60, 120]
# I = [1.0, 2.0, 3.0, 4.0]
I = [0.5, 1.0, 2.5, 3.5, 5.0]
H = max(I)
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -8, -10, -15, -16, -17, -18, -19, -20, -25, -30, -50, -100]
ai = QLearn(actions=levels, epsilon=0.8, alpha=0.5, gamma=0.5)
actionSpace = ActionSpace(orderbook, side, V, T, I, H, ai, levels)
actionSpace_test = ActionSpace(orderbook_test, side, V, T, I, H, ai, levels)

#logging.basicConfig(level=logging.INFO)

# M = actionSpace.runActionBehaviour()
# print(np.asarray(M))

bestExecutions2 = []
avgExecutions2 = []
bestExecutions5 = []
avgExecutions5 = []

for episode in range(100):
    pp.pprint("Episode " + str(episode))
    # # Train
    actionSpace.trainConcurrent(episodes=1)
    np.save('q.npy', actionSpace.ai.q)
    #pp.pprint(actionSpace.ai.q)


    # Backtest
    q = np.load('q.npy').item()
    # M <- [t, i, Price, A, Paid, Diff]
    M = actionSpace_test.backtestConcurrent(q, 1)
    #pp.pprint(M)

    bestExecutions2.append(getBestTimeForInventory(M, 2.5))
    avgExecutions2.append(getAvgPriceDiffForInventory(M, 2.5))
    bestExecutions5.append(getBestTimeForInventory(M, 5.0))
    avgExecutions5.append(getAvgPriceDiffForInventory(M, 5.0))

pp.pprint(avgExecutions2)
pp.pprint(avgExecutions5)
pp.pprint(bestExecutions2)
pp.pprint(bestExecutions5)
