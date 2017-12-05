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
        self.type = type  # MARKET | LIMIT
        self.order = None  # Trade
        self.fills = []   # fills==match(order)

    def getA(self):
        return self.a

    def getOrder(self):
        return self.order

    def getFills(self):
        return self.fills

    def setOrder(self, order):
        self.order = order

    def addFill(self, order):
        self.fills.append(order)

    def addFills(self, orders):
        for order in orders:
            self.addFill(order)


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
        action = Action(a, 'LIMIT')
        action.setOrder(trade)
        return action

    def createLimitActions(self, qty):
        actions = []
        for level in range(self.levels):
            actions.append(self.createLimitAction(qty, level))
        return actions

    def fillLimitAction(self, action):
        matchEngine = MatchEngine(self.orderbook, index=self.index)
        counterTrades, qtyRemain = matchEngine.matchTradeOverTime(action.getOrder())
        action.addFills(counterTrades)
        return action, qtyRemain

    def fillAndMarketLimitAction(self, action):
        action, qtyRemain = self.fillLimitAction(action)
        print("remaining: "+str(qtyRemain))
        if qtyRemain == 0.0:
            return [action]
        print("fill with market order")
        marketActions = actionSpace.createMarketActions(qtyRemain)
        return [action] + marketActions

    def createMarketActions(self, qty):
        actions = []
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
                action = Action(a, 'MARKET')
                action.setOrder(t)
                action.addFill(t)
                actions.append(action)
                qty = 0
            else:
                t = Trade(self.side, amount, price, 0.0)
                action = Action(a, 'MARKET')
                action.setOrder(t)
                action.addFill(t)
                actions.append(action)
                qty = qty - amount

            if qty == 0:
                break

        if qty > 0:
            raise Exception('Not enough liquidity in orderbook state.')
        return actions

    def calculateActionPrice(self, actions):
        # price = 0.0
        # for action in actions:
        #     order = action[1]
        #     price = price + order.getCty() * order.getPrice()
        # return price
        return reduce(
            lambda a, b: a + b.getOrder().getCty() * b.getOrder().getPrice(), actions, 0.0
        )

    def calculateBidAskMidPrice(self, actions):
        return reduce(
            lambda a, b: a + b.getOrder().getCty() * self.state.getBidAskMid(), actions, 0.0
        )

    def calculateActionValue(self, actions):
        actionValue = 0.0
        qty = 0.0
        for action in actions:
            a = action.getA()
            print("action value: " + str(a))
            order = action.getOrder()
            print("market order: " + str(order))
            print("add action value: " + str(a * order.getCty()))
            actionValue = actionValue + a * order.getCty()
            print("with qty share: " + str(order.getCty()))
            qty = qty + order.getCty()
        actionValue = actionValue / qty
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
# print(actionSpace.createMarketActions(1.5))



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
                actions = actionSpace.createMarketActions(remaining) # [(a, trade executed)]
                actionValue = actionSpace.calculateActionValue(actions)
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
                    a = action.getA()
                    print("fill limit with market order")
                    print(action.getOrder())
                    filledActions = actionSpace.fillAndMarketLimitAction(action)
                    print("nr actions resulted out of fill: " + str(len(filledActions)))
                    actionValue = actionSpace.calculateActionValue(filledActions)
                    actionValues.append(actionValue)
                    print("actionValue: " + str(actionValue))
                    actionPrice = actionSpace.calculateActionPrice(filledActions)
                    print("actionPrice: " + str(actionPrice))
                    basePrice = actionSpace.calculateBidAskMidPrice(filledActions)
                    print("basePrice: " + str(basePrice))
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
