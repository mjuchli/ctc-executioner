import logging
import copy
import numpy as np
from qlearn import QLearn
from trade import Trade
from order import Order
from order_type import OrderType
from order_side import OrderSide
from orderbook import Orderbook
from match_engine import MatchEngine
import pprint
import random


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

    def __init__(self, a, runtime):
        self.a = a
        self.runtime = runtime
        self.order = None
        self.trades = []  # filled order

    def getA(self):
        return self.a

    def getRuntime(self):
        return self.runtime

    def setRuntime(self, runtime):
        self.runtime = runtime

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

    def getQtyNotExecuted(self):
        return self.getOrder().getCty() - self.getQtyExecuted()

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
            return midPrice - self.getAvgPrice()
        else:
            return self.getAvgPrice() - midPrice

    def move(self, t_next, i_next):
        newAction = copy.deepcopy(self)
        newAction.setRuntime(t_next)
        newAction.getOrder().setCty(i_next)
        return newAction


class ActionSpace(object):

    def __init__(self, orderbook, side, V, T, I, H, levels=5):
        self.orderbook = orderbook
        self.index = 0
        self.state = None
        self.randomOffset = 0
        self.side = side
        self.levels = range(-levels + 1, levels)
        self.ai = QLearn(self.levels)  # levels are our qlearn actions
        self.V = V
        self.T = T
        self.I = I
        self.H = H

    def setRandomOffset(self):
        # maxOffset = orderbook.getIndexWithTimeRemain(max(self.T))
        # offsetRange = range(maxOffset)
        # self.randomOffset = random.choice(offsetRange)
        self.randomOffset = random.choice(range(1000))

    def getState(self):
        return self.state

    def setState(self, t):
        index = self.orderbook.getIndexWithTimeRemain(t, self.randomOffset)
        if index >= len(self.orderbook.getStates()):
            raise Exception('Index out of orderbook state.')
        self.index = index
        self.state = self.orderbook.getState(self.index)

    def createAction(self, level, runtime, qty, force_execution=False):
        if level <= 0:
            positions = self.state.getSidePositions(self.side)
        else:
            level = level - 1  # 1 -> 0, ect., such that array index fits
            positions = self.state.getSidePositions(self.side.opposite())

        if runtime <= 0.0:
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
        action = Action(a=level, runtime=runtime)
        action.setOrder(order)
        return action

    def createActions(self, runtime, qty, force_execution=False):
        actions = []
        for level in self.levels:
            actions.append(self.createAction(level, runtime, qty, force_execution))
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

    def chooseAction(self, t, i, force_execution=False):
        self.setState(t)  # Important to set orderbook index while matching
        inventory = i * (self.V / max(self.I))
        actions = self.createActions(t, inventory, force_execution)
        for action in actions:
            self.runAction(action, t)
        return self.determineBestAction(actions)

    def demonstrateActionBehaviour(self, episodes=1):
        for episode in range(int(episodes)):
            self.setRandomOffset()
            M = []
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    action = self.chooseAction(t, i, True)
                    state = (t, i)
                    M.append([state, action.getA(), action.getAvgPrice()])
        return M

    def determineNextState(self, action):
        qty_remaining = action.getQtyNotExecuted()
        inventory_remaining = qty_remaining / self.V

        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            inventory_next = inventory_remaining * max(self.I)
            i_next = min([0.0] + self.I, key=lambda x: abs(x - inventory_next))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(inventory_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        if action.getRuntime > 0:
            t_next = T[T.index(action.getRuntime()) - 1]
        else:
            t_next = action.getRuntime()

        logging.info('Next state for action: ' + str((t_next, i_next)))
        return (t_next, i_next)

    def update(self, episodes=1):
        for episode in range(int(episodes)):
            self.setRandomOffset()
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    action = self.chooseAction(t, i, False)
                    (t_next, i_next) = self.determineNextState(action)
                    reference = self.state.getBidAskMid()
                    self.ai.learn(
                        state1=(t, i),
                        action1=action.getA(),
                        reward=(action.getValue(reference)),
                        state2=(t_next, i_next))

                    while i_next != 0:
                        t_old = t_next
                        i_old = i_next
                        inventory_next = i_next * (self.V / max(self.I))
                        #action = action.move(t_next, inventory_next)  # this would proceed with the same level (e.g. a) only
                        action = self.chooseAction(t_next, i_next, False)
                        (t_next, i_next) = self.determineNextState(action)
                        logging.info('next state: ' + str((t_next, i_next)))
                        reference = self.state.getBidAskMid()
                        self.ai.learn(
                            state1=(t_old, i_old),
                            action1=action.getA(),
                            reward=(action.getValue(reference)),
                            state2=(t_next, i_next))



#logging.basicConfig(level=logging.INFO)
orderbook = Orderbook()
orderbook.loadFromFile('query_result.tsv')
side = OrderSide.BUY
V = 1000.0
# T = [4, 3, 2, 1, 0]
T = [0, 1, 2, 5, 10, 30, 60, 120]
# I = [1.0, 2.0, 3.0, 4.0]
I = [1.0, 2.0, 3.0, 4.0]
H = max(I)
actionSpace = ActionSpace(orderbook, side, V, T, I, H)

#M = actionSpace.demonstrateActionBehaviour()
#print(np.asarray(M))

actionSpace.update(1)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(actionSpace.ai.q)
