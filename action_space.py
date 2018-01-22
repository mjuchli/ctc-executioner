import logging
import threading
import numpy as np
from action import Action
from order import Order
from order_type import OrderType
from order_side import OrderSide
from qlearn import QLearn
from action_state import ActionState


class ActionSpace(object):

    def __init__(self, orderbook, side, T, I, ai=None, levels=None):
        self.orderbook = orderbook
        self.side = side
        if not levels:
            levels = [-2, -1, 0, 1, 2, 3]
        self.levels = levels
        if not ai:
            ai = QLearn(self.levels)  # levels are our qlearn actions
        self.ai = ai
        self.T = T
        self.I = I

    def createAction(self, level, t, qty, force_execution=False):
        orderbookState, index = self.orderbook.getRandomState(t, max(self.T))
        aiState = ActionState(t, qty, orderbookState.getMarket())

        if level is None:
            level = self.ai.chooseAction(aiState)
            # print('Random action: ' + str(level) + ' for state: ' + str(aiState))

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
            price = orderbookState.getPriceAtLevel(side, level)

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=qty,
            price=price
        )
        action = Action(a=level, runtime=runtime)
        action.setState(aiState)
        action.setOrder(order)
        action.setOrderbookState(orderbookState)
        action.setOrderbookIndex(index)
        action.setReferencePrice(orderbookState.getBidAskMid())
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
        actions = self.createActions(t, i, force_execution)
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

        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            # Approximate next closest inventory given remaining and I
            i_next = min([0.0] + self.I, key=lambda x: abs(x - qty_remaining))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(qty_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        logging.info('Next inventory for action: ' + str(i_next))
        return i_next

    def update(self, t, i, force_execution=False):
        action = self.createAction(None, t, i, force_execution=force_execution)
        action.run(self.orderbook)
        i_next = self.determineNextInventory(action)
        t_next = self.determineNextTime(t)
        reward = action.getValueAvg(fees=False)
        # reward = action.getValueExecuted()
        # reward = action.getTestReward()
        state_next = ActionState(action.getState().getT(), action.getState().getI(), action.getState().getMarket())
        state_next.setT(t_next)
        state_next.setI(i_next)
        # print("Reward " + str(reward) + ": " + str(action.getState()) + " with " + str(action.getA()) + " -> " + str(state_next))
        self.ai.learn(
            state1=action.getState(),
            action1=action.getA(),
            reward=reward,
            state2=state_next
        )
        return (t_next, i_next)


    def train(self, episodes=1, force_execution=False):
        for episode in range(int(episodes)):
            for t in self.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.I:
                    logging.info("     i=="+str(i))
                    logging.info("Action run " + str((t, i)))
                    (t_next, i_next) = self.update(t, i, force_execution)
                    while i_next != 0:
                        if force_execution:
                            raise Exception("Enforced execution left " + str(i_next) + " unexecuted.")
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))
                        (t_next, i_next) = self.update(t_next, i_next, force_execution)


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
        #T = self.T[1:len(self.T)]
        for t in [self.T[-1]]:
            logging.info("\n"+"t=="+str(t))
            for i in [self.I[-1]]:
                logging.info("     i=="+str(i))
                actions = []
                state = ActionState(t, i, {})
                #print(state)
                try:
                    a = self.ai.getQAction(state, 0)
                    # print("Q action for state " + str(state) + ": " + str(a))
                except:
                    # State might not be in Q-Table yet, more training requried.
                    logging.info("State " + str(state) + " not in Q-Table.")
                    break
                actions.append(a)
                action = self.createAction(a, t, i, force_execution=False)
                midPrice = action.getReferencePrice()

                # print("before...")
                # print("state: " + str(action.getState()))
                # print("runtime: " + str(action.getRuntime()))
                # print("order cty: " + str(action.getOrder().getCty()))
                # print("executed: " + str(action.getQtyExecuted()))
                # print("not executed: " + str(action.getQtyNotExecuted()))
                # print("ref price: " + str(action.getReferencePrice()))
                # print("avg paid: " + str(action.getAvgPrice()))
                # print("order: " + str(action.getOrder()))
                # print("trades: " + str(action.getTrades()))
                # print("reward: " + str(action.getValueAvg()))


                action.run(self.orderbook)
                # print("after...")
                # print("state: " + str(action.getState()))
                # print("runtime: " + str(action.getRuntime()))
                # print("order cty: " + str(action.getOrder().getCty()))
                # print("executed: " + str(action.getQtyExecuted()))
                # print("not executed: " + str(action.getQtyNotExecuted()))
                # print("ref price: " + str(action.getReferencePrice()))
                # print("avg paid: " + str(action.getAvgPrice()))
                # print("order: " + str(action.getOrder()))
                # print("trades: " + str(action.getTrades()))
                # print("reward: " + str(action.getValueAvg()))

                i_next = self.determineNextInventory(action)
                t_next = self.determineNextTime(t)
                # print("i_next: " + str(i_next))
                while i_next != 0:
                    state_next = ActionState(t_next, i_next, {})
                    try:
                        a_next = self.ai.getQAction(state_next, 0)
                        # print("Q action for next state " + str(state_next) + ": " + str(a_next))
                    except:
                        # State might not be in Q-Table yet, more training requried.
                        # print("State " + str(state_next) + " not in Q-Table.")
                        break
                    actions.append(a_next)
                    #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")

                    runtime_next = self.determineRuntime(t_next)
                    action.setState(state_next)
                    action.update(a_next, runtime_next)
                    action.run(self.orderbook)
                    # print("after next...")
                    # print("state: " + str(action.getState()))
                    # print("runtime: " + str(action.getRuntime()))
                    # print("order cty: " + str(action.getOrder().getCty()))
                    # print("executed: " + str(action.getQtyExecuted()))
                    # print("not executed: " + str(action.getQtyNotExecuted()))
                    # print("ref price: " + str(action.getReferencePrice()))
                    # print("avg paid: " + str(action.getAvgPrice()))
                    # print("order: " + str(action.getOrder()))
                    # print("trades: " + str(action.getTrades()))
                    # print("reward: " + str(action.getValueAvg()))

                    # i = i_next
                    # t = t_next
                    i_next = self.determineNextInventory(action)
                    t_next = self.determineNextTime(t_next)

                price = action.getAvgPrice()
                # TODO: last column is for for the BUY scenario only
                if action.getOrder().getSide() == OrderSide.BUY:
                    profit = midPrice - price
                else:
                    profit = price - midPrice
                Ms.append([state, midPrice, actions, price, profit])
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
