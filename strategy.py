import logging
import numpy as np
from qlearn import QLearn
from trade import Trade
from order import Order
from order_type import OrderType
from order_side import OrderSide
from orderbook import Orderbook
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

    def __init__(self, a):
        self.a = a
        self.order = None
        self.trades = []  # filled order

    def getA(self):
        return self.a

    def getOrder(self):
        return self.order

    def setOrder(self, order):
        self.order = order

    def getTrades(self):
        return self.trades

    def setTrades(self, trades):
        self.trades = trades

    def getAvgPrice(self):
        """Returns the average price paid for the executed order."""
        if self.getQtyExecuted() == 0:
            return 0.0

        price = 0.0
        for trade in self.getTrades():
            price = price + trade.getCty() * trade.getPrice()
        return price / self.getQtyExecuted()

    def getQtyExecuted(self):
        qty = 0.0
        for trade in self.getTrades():
            qty = qty + trade.getCty()
        return qty

    def isFilled(self):
        return self.getQtyExecuted() == self.order.getCty()

    def getTotalPaidReceived(self):
        return self.getAvgPrice() * self.getQtyExecuted()

    def getValue(self, midPrice):
        """Retuns price difference to bid/ask-mid. The lower, the better.

        For BUY: total paid - total paid at mid price
        For SELL: total received at mid price - total received
        """
        if self.getOrder().getSide() == OrderSide.BUY:
            if self.getAvgPrice() == 0.0:
                return midPrice
            return self.getAvgPrice() - midPrice
        else:
            return midPrice - self.getAvgPrice()


class ActionSpace(object):

    def __init__(self, orderbook, side, levels=3):
        self.orderbook = orderbook
        self.index = 0
        self.state = None
        self.initialState()
        self.side = side
        self.levels = range(-levels + 1, levels)
        self.ai = QLearn(self.levels)  # levels are our qlearn actions

    def initialState(self):
        self.index = 0
        if not self.orderbook.getStates():
            raise Exception('No states possible, provided orderbook is empty.')
        self.state = self.orderbook.getState(self.index)

    def hasNextState(self):
        if self.index+1 >= len(self.orderbook.getStates()):
            return False
        return True

    def nextState(self):
        self.index = self.index + 1
        if self.index >= len(self.orderbook.getStates()):
            raise Exception('Index out of orderbook state.')
        self.state = self.orderbook.getState(self.index)

    # def createMarketAction(self, qty):
    #     order = Order(orderType=OrderType.MARKET, orderSide=self.side, cty=qty, price=None)
    #     action = Action(None)
    #     action.setOrder(order)
    #     return action

    def createAction(self, level, time, qty, force_execution=False):
        if level <= 0:
            positions = self.state.getSidePositions(self.side)
        else:
            level = level - 1  # 1 -> 0, ect., such that array index fits
            positions = self.state.getSidePositions(self.side.opposite())

        if time <= 0.0:
            price = None
            ot = OrderType.MARKET
        else:
            price = positions[abs(level)].getPrice()
            if force_execution:
                ot = OrderType.LIMIT_T_MARKET
            else:
                ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=qty,
            price=price
        )
        action = Action(level)
        action.setOrder(order)
        return action

    def createActions(self, time, qty, force_execution=False):
        actions = []
        for level in self.levels:
            actions.append(self.createAction(level, time, qty, force_execution))
        return actions

    def runAction(self, action, t):
        matchEngine = MatchEngine(self.orderbook, index=self.index)
        counterTrades, qtyRemain = matchEngine.matchOrder(action.getOrder(), t)
        action.setTrades(counterTrades)
        return action, qtyRemain

    def determineBestAction(self, actions):
        reference = self.state.getBidAskMid()
        bestAction = None
        for action in actions:
            if not bestAction:
                bestAction = action
                continue
            if action.getValue(reference) < bestAction.getValue(reference):
                bestAction = action
        return bestAction

    def chooseAction(self, t, i, V, I, force_execution=False):
        inventory = i * (V / max(I))
        actions = self.createActions(t, inventory, force_execution)
        for action in actions:
            self.runAction(action, t)
        return self.determineBestAction(actions)



#logging.basicConfig(level=logging.INFO)
orderbook = Orderbook()
orderbook.loadFromFile('query_result.tsv')
side = OrderSide.BUY
actionSpace = ActionSpace(orderbook, side)
episodes = 1
V = 10000.0
# T = [4, 3, 2, 1, 0]
T = [0, 1, 2, 5, 10, 30, 60, 120]
# I = [1.0, 2.0, 3.0, 4.0]
I = [1.0, 2.0, 3.0, 4.0]
H = max(I)


for episode in range(int(episodes)):
    actionSpace.initialState()
    M = []
    for t in T:
        print("\n"+"t=="+str(t))
        for i in I:
            print("     i=="+str(i))
            reference = actionSpace.state.getBidAskMid()

            state = (t, i)
            action = actionSpace.chooseAction(t, i, V, I)
            M.append([state, action.getA(), action.getAvgPrice()])
            actionSpace.ai.learn(
                state1=state,
                action1=action.getA(),
                reward=(action.getValue(reference) * -1),
                state2=state
            )
            #print("action: " + str(action))
            # match
            # reward
            # learn
            # ---

            #bidAskMid = actionSpace.state.getBidAskMid()
            #(bestA, bestActionValue, bestPrice) = actionSpace.chooseOptimalAction(t, i, V, H)
            #M.append([t, i, bestA, bestActionValue, bidAskMid, bestPrice])
    print(np.asarray(M))


print(actionSpace.ai.q)
