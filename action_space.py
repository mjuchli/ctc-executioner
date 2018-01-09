import logging
import threading
import numpy as np
from action import Action
from order import Order
from order_type import OrderType
from qlearn import QLearn


class ActionSpace(object):

    def __init__(self, orderbook, side, V, T, I, H, ai=None, levels=None):
        self.orderbook = orderbook
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

    def createAction(self, level, runtime, qty, force_execution=False):
        state, index = self.orderbook.getRandomState(runtime, max(self.T))
        if level <= 0:
            side = self.side
        else:
            level = level - 1  # 1 -> 0, ect., such that array index fits
            side = self.side.opposite()

        if runtime <= 0.0:
            price = None
            ot = OrderType.MARKET
        else:
            price = state.getPriceAtLevel(side, level)
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
        action.setOrderbookState(state)
        action.setOrderbookIndex(index)
        return action

    def createActions(self, runtime, qty, force_execution=False):
        actions = []
        for level in self.levels:
            actions.append(self.createAction(level, runtime, qty, force_execution))
        return actions

    def determineBestAction(self, actions):
        bestAction = None
        for action in actions:
            if not bestAction:
                bestAction = action
                continue
            if action.getValueAvg() < bestAction.getValueAvg():
                bestAction = action
        return bestAction

    def getMostRewardingAction(self, t, i, force_execution=False):
        inventory = i * (self.V / max(self.I))
        actions = self.createActions(t, inventory, force_execution)
        for action in actions:
            action.run(self.ororderbook)
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

    def update(self, aiState, force_execution=False):
        a = self.ai.chooseAction(aiState)
        t = aiState[0]
        i = aiState[1]
        inventory = i * (self.V / max(self.I))
        action = self.createAction(a, t, inventory, force_execution=force_execution)
        action.run(self.orderbook)
        (t_next, i_next) = self.determineNextState(action)
        reward = action.getValueAvg()
        self.ai.learn(
            state1=(t, i),
            action1=action.getA(),
            reward=(reward),
            state2=(t_next, i_next))
        return (t_next, i_next)

    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    logging.info("Action run " + str((t, i)))
                    (t_next, i_next) = self.update((t, i), force_execution)
                    while i_next != 0:
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))
                        (t_next, i_next) = self.update((t_next, i_next), force_execution)

    def runActionBehaviour(self, episodes=1):
        for episode in range(int(episodes)):
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
                action.run(self.orderbook)
                price = action.getAvgPrice()
                midPrice = action.getOrderbookState().getBidAskMid()
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
