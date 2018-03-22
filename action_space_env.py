import logging
from action import Action
from order import Order
from order_type import OrderType

class ActionSpace(object):

    def __init__(self, orderbook, side, T, I, ai=None, levels=None):
        self.orderbook = orderbook
        self.side = side
        self.levels = levels
        self.T = T
        self.I = I

    def getRandomOrderbookState(self):
        return self.orderbook.getRandomState(max(self.T))

    def createAction(self, level, state, orderbookIndex=None, force_execution=False):
        # Determines whether to run and force execution of given t, or if
        # segmentation of t into multiple runtimes is allowed.
        if force_execution:
            runtime = state.getT()
            ot = OrderType.LIMIT_T_MARKET
        else:
            runtime = self.determineRuntime(state.getT())
            ot = OrderType.LIMIT

        if orderbookIndex is None:
            orderbookState, orderbookIndex = self.getRandomOrderbookState()
        else:
            orderbookState = self.orderbook.getState(orderbookIndex)

        if runtime <= 0.0 or level is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = orderbookState.getPriceAtLevel(self.side, level)

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=state.getI(),
            price=price
        )
        action = Action(a=level, runtime=runtime)
        action.setState(state)
        action.setOrder(order)
        action.setOrderbookState(orderbookState)
        action.setOrderbookIndex(orderbookIndex)
        action.setReferencePrice(orderbookState.getBestAsk())
        return action

    def updateAction(self, action, level, state, orderbookIndex=None, force_execution=False):
        if force_execution:
            runtime = state.getT()
            ot = OrderType.LIMIT_T_MARKET
        else:
            runtime = self.determineRuntime(state.getT())
            ot = OrderType.LIMIT

        if orderbookIndex is not None:
            orderbookState = self.orderbook.getState(orderbookIndex)
            action.setOrderbookState(orderbookState)
            action.setOrderbookIndex(orderbookIndex)

        if runtime <= 0.0 or level is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = action.getOrderbookState().getPriceAtLevel(self.side, level)

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=state.getI(),
            price=price
        )
        action.setState(state)
        action.setOrder(order)
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
