import logging
import numpy as np
from ctc_executioner.action_space import ActionSpace
from ctc_executioner.qlearn import QLearn
from ctc_executioner.order_side import OrderSide
from ctc_executioner.orderbook import Orderbook
from ctc_executioner.action_state import ActionState
from ctc_executioner.agent_utils.ui import UI
import pprint
import datetime
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


def run_profit(epochs_train=10, epochs_test=5, fixed_a=None):
    if epochs_train > 0:
        q = train(epochs_train)
    M = test(epochs_test, average=False, fixed_a=fixed_a)
    M = np.array(M)
    # print(M)
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


def evaluateReturns(levels=range(-100, 101), crossval=10, force_execution=True, trade_log=False):
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
            if trade_log:
                print("\nLEVEL: " + str(level))
                print("-----------")
                print("Reference price: " + str(refBefore) + " ("+str(action.getOrderbookState().getTimestamp())+", index="+str(action.getOrderbookIndex())+")")
            action.run(actionSpace.orderbook)
            refAfter = action.getOrderbookState().getTradePrice()
            paid = action.getAvgPrice()
            if trade_log:
                print("Order: " + str(action.getOrder()))
                print("Trades:")
                print(action.getTrades())
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


def reject_outliers(data, m=1.5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def priceReturnCurve(enable_after_exec_return=True, levels=range(-100, 101), crossval=10, force_execution=True, filter_outliers=False, trade_log=False):
    (x, ys, ys2) = evaluateReturns(levels, crossval, force_execution, trade_log)
    if filter_outliers:
        y = [np.mean(reject_outliers(np.array(x))) for x in ys]
        y2 = [np.mean(reject_outliers(np.array(x))) for x in ys2]
    else:
        y = [np.mean(np.array(x)) for x in ys]
        y2 = [np.mean(np.array(x)) for x in ys2]

    plt.plot(x, y, 'r-')
    if enable_after_exec_return:
        plt.plot(x, y2, 'g-')
    plt.grid(linestyle='-', linewidth=2)
    plt.show()

#logging.basicConfig(level=logging.DEBUG)

side = OrderSide.BUY
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -10, -12, -15]
ai = QLearn(actions=levels, epsilon=0.4, alpha=0.3, gamma=0.8)

#trainBook = 'query_result_train_15m.tsv'
#testBook = 'query_result_train_15m.tsv'

# orderbook = Orderbook(extraFeatures=False)
# orderbook.loadFromBitfinexFile('orderbook_bitfinex_btcusd_view.tsv')
# orderbook_test = Orderbook(extraFeatures=False)
# orderbook_test.loadFromBitfinexFile('orderbook_bitfinex_btcusd_view.tsv')

# Load orderbook
cols = ["ts", "seq", "size", "price", "is_bid", "is_trade", "ttype"]
import pandas as pd
events = pd.read_table('data/events/ob-1-small.tsv', sep='\t', names=cols, index_col="seq")
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


T = [0, 10, 20, 40, 60, 80, 100] #, 120, 240]
T_test = [0, 10, 20, 40, 60, 80, 100]# 120, 240]

I = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
actionSpace = ActionSpace(orderbook, side, T, I, ai, levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, ai, levels)

#priceReturnCurve(crossval=1)


UI.animate(run_profit, interval=100)
# UI.animate(run_q_reward, interval=1000)
