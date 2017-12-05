import numpy as np
from qlearn import QLearn
from trade import Trade
from order_type import OrderType
from orderbook import Orderbook, OrderbookEntry, OrderbookState
from match_engine import MatchEngine


class Strategy(object):

    def __init__(self):
        self.reward_table = np.array([
            [0,     0],
            [1,     0.625],
            [0.5,   1.25],
            [0.625, 2.5],
            [1.25,  5],
            [0,     0]]
        ).astype("float32")
        self.ai = QLearn(actions=[-1, 1], epsilon=0.3, alpha=0.5, gamma=0.5)
        self.states = [0, 1, 2, 3, 4, 5]
        self.lastState = None
        self.lastAction = None

    def getReward(self, state, action):
        if action == 1:
            i = 1
        else:
            i = 0
        return(self.reward_table[state, i])

    def goal(self):
        return not(self.lastState > 0 and self.lastState < 5)

    def resetState(self):
        random_state = np.random.RandomState(1999)
        states = self.states
        random_state.shuffle(states)
        self.lastState = states[0]

    def update(self):
        if not self.lastState:
            self.resetState()

        action = self.ai.chooseAction(self.lastState)
        state = self.lastState + action
        reward = self.getReward(self.lastState, action)
        self.ai.learn(self.lastState, action, reward, state)
        self.lastState = state
        self.lastAction = action

class Action(object):

    def __init__(self, a, type):
        self.a = a
        self.type = type
        self.orders = []
        self.fills = []

    def addOrder(self, order):
        self.orders.append(order)

    def addFill(self, order):
        self.fills.append(order)


class ActionSpace(object):

    def __init__(self, orderbook, side, levels=3):
        self.orderbook = orderbook
        self.index = 0
        self.state = self.orderbook.getState(self.index)
        self.side = side
        self.levels = levels
        self.qty = 0

    def setIndex(self, index):
        self.index = index

    def getBookPositions(self, side):
        if side == OrderType.BUY:
            return self.state.getBuyers()
        elif side == OrderType.SELL:
            return self.state.getSellers()

    def getBasePrice(self):
        return self.getBookPositions(self.side)[0].getPrice()

    def createLimitAction(self, qty, level):
        basePrice = self.getBasePrice()
        positions = self.getBookPositions(self.side)
        price = positions[level].getPrice()
        if self.side == OrderType.BUY:
            a = price - basePrice
        else:
            a = basePrice - price
        trade = Trade(self.side, qty, price, 0.0)
        return (a, trade)

    def createLimitActions(self, qty):
        actions = []
        for level in range(self.levels):
            actions.append(self.createLimitAction(qty, level))
        return actions

    def createMarketAction(self, qty):
        trades = []
        positions = self.getBookPositions(self.side.opposite())
        basePrice = self.getBasePrice()
        for p in positions:
            price = p.getPrice()
            amount = p.getQty()
            if self.side == OrderType.BUY:
                a = price - basePrice
            else:
                a = basePrice - price

            if amount >= qty:
                t = Trade(self.side, qty, price, 0.0)
                trades.append((a, t))
                qty = 0
            else:
                t = Trade(self.side, amount, price, 0.0)
                trades.append((a, t))
                qty = qty - amount

            if qty == 0:
                break

        if qty > 0:
            raise Exception('Not enough liquidity in orderbook state.')
        return trades

    def calculateActionPrice(self, actions):
        # price = 0.0
        # for action in actions:
        #     order = action[1]
        #     price = price + order.getCty() * order.getPrice()
        # return price
        return reduce(
            lambda a, b: a + b[1].getCty() * b[1].getPrice(), actions, 0.0
        )

    def calculateBidAskMidPrice(self, actions):
        return reduce(
            lambda a, b: a + b[1].getCty() * self.state.getBidAskMid(), actions, 0.0
        )

    def calculateMarketActionValue(self, actions):
        actionValue = 0
        for action in actions:
            a = action[0]
            print("action value: " + str(a))
            order = action[1]
            print("market order: " + str(order))
            print("add action value: " + str(a * order.getCty()))
            actionValue = actionValue + a * order.getCty()
        actionValue = actionValue / remaining
        return actionValue

    def calculateLimitActionValue(self, action):
        a = action[0]
        print("action value: " + str(a))
        order = action[1]
        print("limit order: " + str(order))
        matchEngine = MatchEngine(orderbook, index=self.index)
        counterTrades, qtyRemain = matchEngine.matchTradeOverTime(order)
        print("all countertrades: " + str(counterTrades))
        print("remaining: "+str(qtyRemain))
        if qtyRemain == 0.0:
            actionValue = a
        else:
            actionValue = 0
            for ct in counterTrades:
                print("add actionValue: " + str(a * ct.getCty()))
                actionValue = actionValue + a * ct.getCty()
            print("actionValue total: " + str(actionValue))
            print("qty total: " + str(remaining))
            actionValue = actionValue / remaining
            marketActions = actionSpace.createMarketAction(qtyRemain)
            actionValue = actionValue + actionSpace.calculateMarketActionValue(marketActions)
        return actionValue


s1 = OrderbookState(1.0)
s1.addBuyers([
    OrderbookEntry(price=0.9, qty=1.5),
    OrderbookEntry(price=0.8, qty=1.0),
    OrderbookEntry(price=0.7, qty=2.0)
    ])
s1.addSellers([
    OrderbookEntry(price=1.1, qty=1.0),
    OrderbookEntry(price=1.2, qty=1.0),
    OrderbookEntry(price=1.3, qty=3.0)
    ])

s2 = OrderbookState(1.1)
s2.addBuyers([
    OrderbookEntry(price=1.0, qty=1.5),
    OrderbookEntry(price=0.9, qty=1.0),
    OrderbookEntry(price=0.8, qty=2.0)
    ])
s2.addSellers([
    OrderbookEntry(price=1.2, qty=1.0),
    OrderbookEntry(price=1.3, qty=1.0),
    OrderbookEntry(price=1.4, qty=3.0)
    ])

orderbook = Orderbook()
orderbook.addState(s1)
orderbook.addState(s2)
# orderbook.addState(s3)

# actionSpace = ActionSpace(orderbook, OrderType.BUY)
# actionSpace.orderbookState = orderbook[0]
# actions = actionSpace.getLimitOrders(qty=1)
# print(actions)
# print(actionSpace.createMarketAction(1.5))



side = OrderType.BUY
s = Strategy()
actionSpace = ActionSpace(orderbook, side)
episodes = 1
V = 4.0
# T = [4, 3, 2, 1, 0]
T = [0, 1, 2]
# I = [1.0, 2.0, 3.0, 4.0]
I = [1.0, 2.0, 3.0, 4.0]


for episode in range(int(episodes)):
    s.resetState()
    M = []
    for t in T:
        o = 0
        print("\n"+"t=="+str(t))
        # while len(orderbook) > o:
        # print("observe orderbook with state: " + str(o))
        # orderbook -> o{}
        for i in I:
            remaining = i*(V/max(I))
            print("remaining inventory: " + str(remaining))
            orderbookState = orderbook.getState(o)
            if t == 0:
                print("time consumed: market order")
                actions = actionSpace.createMarketAction(remaining) # [(a, trade executed)]
                actionValue = actionSpace.calculateMarketActionValue(actions)
                print("actionValue: " + str(actionValue))
                actionPrice = actionSpace.calculateActionPrice(actions)
                print("actionPrice: " + str(actionPrice))
                basePrice = actionSpace.calculateBidAskMidPrice(actions)
                print("basePrice: " + str(basePrice))

            else:
                print("time left: limit order")
                actionSpace.orderbookState = orderbookState
                actions = actionSpace.createLimitActions(remaining) # [(a, trade unexecuted)]
                actions
                print("actions:"+str(actions)+"\n")
                actionValues = []
                for action in actions:
                    a = action[0]
                    trade = action[1]
                    #executedTrades =
                    actionValue = actionSpace.calculateLimitActionValue(action)
                    actionValues.append(actionValue)
                    print("actionValue: " + str(actionValue))
                    print("")
                actionValue = min(actionValues)
            M.append([t, i, actionValue])
    print(np.asarray(M))
            #     L = s.getPossibleActions()
            #     for a in L:
            #         s.update()
            #         # s.lastState
            #         # s.lastAction
            #         # s.ai.q

            # o = o + 1

print(s.ai.q)
