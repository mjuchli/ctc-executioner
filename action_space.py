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

    def createAction(self, level, t, qty, force_execution=False):
        state, index = self.orderbook.getRandomState(t, max(self.T))

        # Determines whether to run and force execution of given t, or if
        # segmentation of t into multiple runtimes is allowed.
        if force_execution:
            runtime = t
            ot = OrderType.LIMIT_T_MARKET
        else:
            runtime = self.determineRuntime(t)
            ot = OrderType.LIMIT

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

    def determineRuntime(self, t):
        if t != 0:
            T_index = self.T.index(t)
            runtime = self.T[T_index] - self.T[T_index - 1]
        else:
            runtime = t
        return runtime

    def determineNextTime(self, t):
        if t > 0:
            t_next = self.T[self.T.index(t) - 1]
        else:
            t_next = t

        logging.info('Next timestep for action: ' + str(t_next))
        return t_next

    def determineNextInventory(self, action):
        qty_remaining = action.getQtyNotExecuted()
        inventory_remaining = qty_remaining / self.V

        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            inventory_next = inventory_remaining * max(self.I)
            # Approximate next closest inventory given remaining and I
            i_next = min([0.0] + self.I, key=lambda x: abs(x - inventory_next))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(inventory_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        logging.info('Next inventory for action: ' + str(i_next))
        return i_next

    def update(self, aiState, force_execution=False):
        a = self.ai.chooseAction(aiState)
        t = aiState[0]
        i = aiState[1]
        inventory = i * (self.V / max(self.I))
        action = self.createAction(a, t, inventory, force_execution=force_execution)
        action.run(self.orderbook)
        i_next = self.determineNextInventory(action)
        t_next = self.determineNextTime(t)
        #reward = action.getValueAvg()
        reward = action.getValueExecuted()
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


    # def train(self, episodes=1, force_execution=False):
    #     for episode in range(int(episodes)):
    #         for t in self.T:
    #             logging.info("\n"+"t=="+str(t))
    #             for i in self.I:
    #                 logging.info("     i=="+str(i))
    #                 aiState = (t, i)
    #                 (t_next, i_next) = self.update((t, i), force_execution)
    #                 a = self.ai.chooseAction(aiState)
    #                 inventory = i * (self.V / max(self.I))
    #                 action = self.createAction(a, t, inventory, force_execution=force_execution)
    #                 logging.info("Action run " + str(aiState))
    #                 action.run(self.orderbook)
    #                 i_next = self.determineNextInventory(action)
    #                 t_next = self.determineNextTime(t)
    #                 reward = action.getValueAvg()
    #                 self.ai.learn(
    #                     state1=(t, i),
    #                     action1=action.getA(),
    #                     reward=(reward),
    #                     state2=(t_next, i_next))
    #                 while i_next != 0:
    #                     runtime_next = self.determineRuntime(t_next)
    #                     aiState_next = (t_next, i_next)
    #                     logging.info("Action transition " + str(aiState) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")
    #                     action.setRuntime(runtime_next)
    #                     action.getOrder().setCty(action.getQtyNotExecuted())
    #                     action.run(self.orderbook)
    #                     reward = action.getValueAvg()
    #                     self.ai.learn(
    #                         state1=(t, i),
    #                         action1=action.getA(),
    #                         reward=(reward),
    #                         state2=(t_next, i_next))
    #                     i = i_next
    #                     t = t_next
    #                     i_next = self.determineNextInventory(action)
    #                     t_next = self.determineNextTime(t_next)


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

    def backtest(self, q=None, episodes=10, average=False):
        if q is None:
            q = self.ai.q
        else:
            self.ai.q = q

        if not q:
            raise Exception('Q-Table is empty, please train first.')

        Ms = []
        T = self.T[1:len(self.T)]
        for t in T:
            logging.info("\n"+"t=="+str(t))
            for i in self.I:
                logging.info("     i=="+str(i))
                actions = []
                state = (t, i)
                #print(state)
                #try:
                a = self.ai.getQAction(state)
                #except:
                #    continue
                actions.append(a)
                inventory = i * (self.V / max(self.I))
                action = self.createAction(a, t, inventory, force_execution=False)
                action.run(self.orderbook)
                i_next = self.determineNextInventory(action)
                t_next = self.determineNextTime(t)
                while i_next != 0:
                    state_next = (t_next, i_next)
                    try:
                        a_next = self.ai.getQAction(state_next)
                    except:
                        break
                    actions.append(a_next)
                    #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")

                    runtime_next = self.determineRuntime(t_next)
                    action.update(a_next, runtime_next)
                    action.run(self.orderbook)
                    # i = i_next
                    # t = t_next
                    i_next = self.determineNextInventory(action)
                    t_next = self.determineNextTime(t_next)

                price = action.getAvgPrice()
                midPrice = action.getOrderbookState().getBidAskMid()
                # TODO: last column is for for the BUY scenario only
                Ms.append([state, midPrice, actions, price, midPrice - price])
        if not average:
            return Ms
        return self.averageBacktest(Ms)

    def averageBacktest(self, M):
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
