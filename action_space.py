import logging
import random
import threading
import numpy as np
from action import Action
from match_engine import MatchEngine
from order import Order
from order_type import OrderType
from qlearn import QLearn


class ActionSpace(object):

    def __init__(self, orderbook, side, V, T, I, H, ai=None, levels=None):
        self.orderbook = orderbook
        self.index = 0
        self.state = None
        self.randomOffset = 0
        self.side = side
        if not levels:
            levels = [-2, -1, 0, 1, 2, 3]
        self.levels = levels
        if not ai:
            ai = QLearn(self.levels)  # levels are our qlearn actions
        self.ai = ai
        self.V = V
        self.T = T
        self.I = I
        self.H = H

    def setRandomOffset(self):
        self.randomOffset = self.orderbook.getRandomOffset(max(T))

    def getState(self):
        return self.state

    def setState(self, t):
        index = self.orderbook.getIndexWithTimeRemain(t, self.randomOffset)
        if index >= len(self.orderbook.getStates()):
            raise Exception('Index out of orderbook state.')
        self.index = index
        self.state = self.orderbook.getState(self.index)

    def createAction(self, level, runtime, qty, force_execution=False):
        self.setState(runtime)  # Important to set orderbook index beforehand
        if level <= 0:
            side = self.side
        else:
            level = level - 1  # 1 -> 0, ect., such that array index fits
            side = self.side.opposite()

        if runtime <= 0.0:
            price = None
            ot = OrderType.MARKET
        else:
            price = self.state.getPriceAtLevel(side, level)
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

    def runAction(self, action, break_time_period=False):
        self.setState(action.getRuntime())  # Important to set orderbook index while matching
        matchEngine = MatchEngine(self.orderbook, index=self.index)
        if not break_time_period:
            counterTrades, qtyRemain = matchEngine.matchOrder(action.getOrder(), action.getRuntime())
        else:
            raise Exception("todo")

        action.setTrades(counterTrades)
        return action, qtyRemain

    def determineBestAction(self, actions):
        reference = self.state.getBidAskMid()
        bestAction = None
        for action in actions:
            if not bestAction:
                bestAction = action
                continue
            if action.getValueAvg(reference) < bestAction.getValueAvg(reference):
                bestAction = action
        return bestAction

    def getMostRewardingAction(self, t, i, force_execution=False):
        inventory = i * (self.V / max(self.I))
        actions = self.createActions(t, inventory, force_execution)
        for action in actions:
            self.runAction(action)
        return self.determineBestAction(actions)

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

        if action.getRuntime() > 0:
            t_next = self.T[self.T.index(action.getRuntime()) - 1]
        else:
            t_next = action.getRuntime()

        logging.info('Next state for action: ' + str((t_next, i_next)))
        return (t_next, i_next)

    def update(self, aiState):
        a = self.ai.chooseAction(aiState)
        t = aiState[0]
        i = aiState[1]
        inventory = i * (self.V / max(self.I))
        #action = self.createAction(a, t, inventory, force_execution=False)
        action = self.createAction(a, t, inventory, force_execution=True)
        self.runAction(action)
        (t_next, i_next) = self.determineNextState(action)
        reference = self.state.getBidAskMid()
        reward = action.getValueAvg(reference)
        self.ai.learn(
            state1=(t, i),
            action1=action.getA(),
            reward=(reward),
            state2=(t_next, i_next))
        return (t_next, i_next)

    def train(self, t, episodes=1):
        for episode in range(int(episodes)):
            self.setRandomOffset()
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    (t_next, i_next) = self.update((t, i))
                    while i_next != 0:
                        print("should not happen. state: " + str((t, i)))
                        (t_next, i_next) = self.update((t_next, i_next))

    def trainConcurrent(self, episodes=1):
        threads = []
        for episode in range(int(episodes)):
            t = threading.Thread(target=self.train, args=(1,))
            threads.append(t)

        [t.start() for t in threads]
        [t.join() for t in threads]


    def runActionBehaviour(self, episodes=1):
        for episode in range(int(episodes)):
            self.setRandomOffset()
            M = []
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    action = self.getMostRewardingAction(t, i, True)
                    state = (t, i)
                    M.append([state, action.getA(), action.getAvgPrice()])
        return M

    def backtest(self, q=None, M=[]):
        if q is None:
            q = self.ai.q
        if not q:
            raise Exception('Q-Table is empty, please train first.')

        self.setRandomOffset()
        for t in self.T:
            logging.info("\n"+"t=="+str(t))
            for i in self.I:
                logging.info("     i=="+str(i))
                state = (t, i)
                values = [q.get((state, x)) for x in list(reversed(self.levels))]
                maxQ = max(list(filter(None, values)))
                a = list(reversed(self.levels))[values.index(maxQ)]
                inventory = i * (self.V / max(self.I))
                action = self.createAction(a, t, inventory, force_execution=True)
                self.runAction(action)
                price = action.getAvgPrice()
                self.setState(t)
                midPrice = self.state.getBidAskMid()
                # TODO: last column is for for the BUY scenario only
                M.append([state, midPrice, a, price, midPrice - price])
        return M

    def backtestConcurrent(self, q=None, episodes=1):
        threads = []
        M = []
        for episode in range(int(episodes)):
            t = threading.Thread(target=self.backtest, args=(q, M,))
            threads.append(t)
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Average states within M
        N = []
        observed = []
        for x in M:
            state = x[0]
            if state in observed:
                continue
            observed.append(state)
            paid = []
            reward = []
            for y in M:
                if y[0] == state:
                    paid.append(y[3])
                    reward.append(y[4])
            N.append([state, x[1], x[2], np.average(paid), np.average(reward)])
        return N
