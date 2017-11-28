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
    def __init__(self, side, levels=3):
        self.side = side
        self.levels = levels
        self.orderbookState = None

    def oppositeSide(self):
        if self.side == OrderType.BUY:
            return OrderType.SELL
        if self.side == OrderType.SELL:
            return OrderType.BUY

    def getOrders(self, side):
        if side == OrderType.BUY:
            a = self.orderbookState[1]
        elif side == OrderType.SELL:
            a = self.orderbookState[0]
        return a

    def getBasePrice(self):
        return self.getOrders(self.side)[0][0]

    def getActions(self):
        return(self.getSideActions(self.side)
               + self.getSideActions(self.oppositeSide()))

    def getSideActions(self, side):
        actions = []
        basePrice = self.getBasePrice()
        orders = self.getOrders(side)
        for i in range(self.levels):
            price = orders[i][0]
            if self.side == OrderType.BUY:
                a = price - basePrice
            else:
                a = basePrice - price
            actions.append((a, price))
        return actions


orderbook = [
        # state 1
        [
            # sellers
            [
                [1.1, 1],
                [1.2, 1],
                [1.3, 1]
            ],
            # buyers
            [
                [0.9, 1],
                [0.8, 1],
                [0.7, 1]
            ]
        ],
        # state 2
        [
            # sellers
            [
                [1.2, 1],
                [1.3, 1],
                [1.4, 1]
            ],
            # buyers
            [
                [1.0, 1],
                [0.9, 1],
                [0.8, 1]
            ]
        ]
    ]


def getBidAskMid(orderbookState):
    return (orderbookState[0][0][0] + orderbookState[1][0][0]) / 2.0


def executeMarket(side, qty, orderbookState):
    trades = []
    if side == OrderType.BUY:
        book = orderbookState[0]
    elif side == OrderType.SELL:
        book = orderbookState[1]

    for p in book:
        price = p[0]
        amount = p[1]
        if amount >= qty:
            t = Trade(side, qty, price, 0.0)
            trades.append(t)
            qty = 0
            break
        else:
            t = Trade(side, amount, price, 0.0)
            trades.append(t)
            qty = qty - amount

    if qty > 0:
        raise Exception('Not enough liquidity in orderbook state.')
    return trades


def executeLimit(side, qty, price):
    return Trade(side, qty, price, 0.0)


side = OrderType.SELL
s = Strategy()
actionSpace = ActionSpace(side)
episodes = 45000
V = 2.0
T = [4, 3, 2, 1, 0]
I = [1.0, 2.0, 3.0, 4.0]
for episode in range(int(episodes)):
    bidAskMid = getBidAskMid(orderbook[0])
    bidAskMid

    s.resetState()
    for t in T:
        o = 0
        while len(orderbook) > o:
            # orderbook -> o{}
            for i in I:
                i = 2.0
                remaining = i*(V/max(I))
                orderbookState = orderbook[o]
                if t == 0:
                    tradeActions = executeMarket(side, remaining, orderbookState)
                    tradeActions
                    #reward = avg(a of trades) - bidAskMid
                else:
                    actionSpace.orderbookState = orderbookState
                    actions = actionSpace.getActions()
                    actions
                    tradeActions = []
                    for action in actions:
                        a = action[0]
                        price = action[1]
                        tradeActions.append((a, executeLimit(side, o, actions)))
                        tradeActions
                    tradeActions
                for trade in trades:
                    print(trade)

            #     L = s.getPossibleActions()
            #     for a in L:
            #         s.update()
            #         # s.lastState
            #         # s.lastAction
            #         # s.ai.q
            o = o + 1

print(s.ai.q)
