import numpy as np
from qlearn import QLearn
from trade import Trade
from order_type import OrderType

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


class ActionSpace(object):
    def __init__(self, orderbook, side, levels=3):
        self.orderbook = orderbook
        self.index = 0
        self.state = self.orderbook[self.index]
        self.side = side
        self.levels = levels
        self.qty = 0

    def setIndex(self, index):
        self.index = index

    def getBookPositions(self, side):
        if side == OrderType.BUY:
            a = self.state[1]
        elif side == OrderType.SELL:
            a = self.state[0]
        return a

    def getBasePrice(self):
        return self.getBookPositions(self.side)[0][0]

    def getBidAskMid(self):
        return (self.state[0][0][0] + self.state[1][0][0]) / 2.0

    def createLimitOrder(self, qty, level):
        basePrice = self.getBasePrice()
        positions = self.getBookPositions(self.side)
        price = positions[level][0]
        if self.side == OrderType.BUY:
            a = price - basePrice
        else:
            a = basePrice - price
        trade = Trade(self.side, qty, price, 0.0)
        return (a, trade)

    def createLimitOrders(self, qty):
        actions = []
        for level in range(self.levels):
            actions.append(self.createLimitOrder(qty, level))
        return actions

    def createMarketOrder(self, qty):
        trades = []
        positions = self.getBookPositions(self.side.opposite())
        basePrice = self.getBasePrice()
        for p in positions:
            price = p[0]
            amount = p[1]
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

    def calculateMarketActionPrice(self, actions):
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
            lambda a, b: a + b[1].getCty() * self.getBidAskMid(), actions, 0.0
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

    def matchTrade(self, trade, orderbookState):
        sellers = orderbookState[0]
        buyers = orderbookState[1]
        totalQty = trade.getCty()
        if trade.getType == OrderType.BUY:
            for p in sellers:
                price = p[0]
                qty = p[1]
                if price == trade.getPrice():
                    if qty >= totalQty:
                        return Trade(OrderType.SELL, totalQty, price)
                    else:
                        # partial execution
                        return Trade(OrderType.SELL, qty, price)
        else:
            for p in buyers:
                price = p[0]
                qty = p[1]
                if price == trade.getPrice():
                    if qty >= totalQty:
                        return Trade(OrderType.BUY, totalQty, price, 0.0)
                    else:
                        # partial execution
                        return Trade(OrderType.BUY, qty, price, 0.0)

    def matchTradeOverTime(self, trade):
        i = self.index
        remaining = trade.getCty()
        trades = []
        while len(orderbook) > i and remaining > 0:
            orderbookState = self.orderbook[i]
            counterTrade = self.matchTrade(trade, orderbookState)
            if counterTrade:
                print("counter trade: " + str(counterTrade))
                remaining = remaining - counterTrade.getCty()
                trade.setCty(remaining)
                trades.append(counterTrade)
            i = i+1
        return trades, remaining

    def calculateLimitActionValue(self, action):
        a = action[0]
        print("action value: " + str(a))
        order = action[1]
        print("limit order: " + str(order))
        counterTrades, qtyRemain = self.matchTradeOverTime(order)
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
            marketActions = actionSpace.createMarketOrder(qtyRemain)
            actionValue = actionValue + actionSpace.calculateMarketActionValue(marketActions)
        return actionValue


class OrderbookEntry(object):

    def __init__(self, price, qty):
        self.price = price
        self.qty = qty

    def __str__(self):
        return str(self.price) + ": " + str(self.qty)

    def __repr__(self):
        return str(self)


class OrderbookState(object):

    def __init__(self, tradePrice=0.0):
        self.tradePrice = tradePrice
        self.buyers = []
        self.sellers = []

    def __str__(self):
        s = ""
        s = s + "Price: " + str(self.tradePrice) + "\n"
        s = s + "Buyers: " + str(self.buyers) + "\n"
        s = s + "Sellers: " + str(self.sellers)
        return s

    def __repr__(self):
        return str(self)

    def setTradePrice(self, tradePrice):
        self.tradePrice = tradePrice

    def addBuyer(self, entry):
        self.buyers.append(entry)

    def addBuyers(self, entries):
        for entry in entries:
            self.buyers.append(entry)

    def addSeller(self, entry):
        self.sellers.append(entry)

    def addSellers(self, entries):
        for entry in entries:
            self.sellers.append(entry)


class Orderbook(object):

    def __init__(self):
        self.states = []

    def __str__(self):
        s = ''
        i = 1
        for state in self.states:
            s = s + 'State ' + str(i) + "\n"
            s = s + '-------' + "\n"
            s = s + str(state)
            s = s + "\n\n"
            i = i + 1
        return s

    def __repr__(self):
        return str(self)

    def addState(self, state):
        self.states.append(state)

    def addStates(self, states):
        for state in states:
            self.states.append(state)


s1 = OrderbookState(1.0)
s1.addBuyers([
    OrderbookEntry(price=0.9, qty=1.5),
    OrderbookEntry(price=0.8, qty=1.0),
    OrderbookEntry(price=0.7, qty=2.0)
    ])
s1.addSellers([
    OrderbookEntry(price=1.1, qty=1.0),
    OrderbookEntry(price=0.2, qty=1.0),
    OrderbookEntry(price=0.3, qty=3.0)
    ])

s2 = OrderbookState(1.1)
s2.addBuyers([
    OrderbookEntry(price=0.9, qty=1.5),
    OrderbookEntry(price=0.8, qty=1.0),
    OrderbookEntry(price=0.7, qty=2.0)
    ])
s2.addSellers([
    OrderbookEntry(price=1.1, qty=1.0),
    OrderbookEntry(price=0.2, qty=1.0),
    OrderbookEntry(price=0.3, qty=3.0)
    ])

ob = Orderbook()
ob.addState(s1)
ob.addState(s2)
# ob.addState(s3)
print(ob)

orderbook = [
        # state 1
        [
            # sellers
            [
                [1.1, 1],
                [1.2, 1],
                [1.3, 3]
            ],
            # buyers
            [
                [0.9, 1.5],
                [0.8, 1],
                [0.7, 2]
            ]
        ],
        # state 2
        [
            # sellers
            [
                [1.2, 1],
                [1.3, 1],
                [1.4, 2]
            ],
            # buyers
            [
                [1.0, 1],
                [0.9, 1.25],
                [0.8, 2]
            ]
        ]
    ]

# actionSpace = ActionSpace(orderbook, OrderType.BUY)
# actionSpace.orderbookState = orderbook[0]
# actions = actionSpace.getLimitOrders(qty=1)
# print(actions)
# print(actionSpace.createMarketOrder(1.5))



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
            orderbookState = orderbook[o]
            if t == 0:
                print("time consumed: market order")
                orders = actionSpace.createMarketOrder(remaining)
                actionValue = actionSpace.calculateMarketActionValue(orders)
                print("actionValue: " + str(actionValue))
                actionPrice = actionSpace.calculateMarketActionPrice(orders)
                print("actionPrice: " + str(actionPrice))
                basePrice = actionSpace.calculateBidAskMidPrice(orders)
                print("basePrice: " + str(basePrice))

            else:
                print("time left: limit order")
                actionSpace.orderbookState = orderbookState
                actions = actionSpace.createLimitOrders(remaining)
                actions
                print("actions:"+str(actions)+"\n")
                actionValues = []
                for action in actions:
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
