import logging
import copy
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from action import Action
from action_state import ActionState
from order import Order
from order_type import OrderType
from order_side import OrderSide

class ExecutionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.side = OrderSide.SELL
        self.levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -9]
        self.T = self._generate_Sequence(min=0, max=500, step=20)
        self.I = self._generate_Sequence(min=0, max=1.0, step=0.1)
        self.lookback = 5 # results in (bid|size, ask|size) -> 4*5
        self.bookSize = 10
        self.orderbookIndex = None
        self.actionState = None
        self.execution = None
        self.action_space = spaces.Discrete(len(self.levels))
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2*self.lookback, self.bookSize, 2))

    def _generate_Sequence(self, min, max, step):
        """ Generate sequence (that unlike xrange supports float)

        max: defines the sequence maximum
        step: defines the interval
        """
        i = min
        I = []
        while i < max:
            I.append(i)
            i = i + step
        return I

    def _configure(self, orderbook):
        self.orderbook = orderbook

    def _determine_next_inventory(self, execution):
        qty_remaining = execution.getQtyNotExecuted()

        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            # Approximate next closest inventory given remaining and I
            i_next = min([0.0] + self.I, key=lambda x: abs(x - qty_remaining))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(qty_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        logging.info('Next inventory for execution: ' + str(i_next))
        return i_next

    def _determine_next_time(self, t):
        if t > 0:
            t_next = self.T[self.T.index(t) - 1]
        else:
            t_next = t

        logging.info('Next timestep for execution: ' + str(t_next))
        return t_next

    def _determine_runtime(self, t):
        if t != 0:
            T_index = self.T.index(t)
            runtime = self.T[T_index] - self.T[T_index - 1]
        else:
            runtime = t
        return runtime

    def _get_random_orderbook_state(self):
        return self.orderbook.getRandomState(max(self.T))

    def _create_execution(self, a):
        runtime = self._determine_runtime(self.actionState.getT())
        orderbookState = self.orderbook.getState(self.orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = orderbookState.getPriceAtLevel(self.side, a)
            ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=self.actionState.getI(),
            price=price
        )
        execution = Action(a=a, runtime=runtime)
        execution.setState(self.actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(self.orderbookIndex)
        execution.setReferencePrice(orderbookState.getBestAsk())
        return execution

    def _update_execution(self, execution, a):
        runtime = self._determine_runtime(self.actionState.getT())
        orderbookState = self.orderbook.getState(self.orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = execution.getOrderbookState().getPriceAtLevel(self.side, a)
            ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=self.actionState.getI(),
            price=price
        )
        execution.setState(self.actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(self.orderbookIndex)
        return execution

    def step(self, action):
        action = self.levels[action]
        if self.execution is None:
            self.execution = self._create_execution(action)
        else:
            self.execution = self._update_execution(self.execution, action)
        self.execution.run(self.orderbook)

        i_next = self._determine_next_inventory(self.execution)
        t_next = self._determine_next_time(self.execution.getState().getT())
        reward = self.execution.getValueAvg(fees=False)
        bidAskFeature = self.orderbook.getBidAskFeatures(
            self.execution.getOrderbookIndex(),
            lookback=self.lookback,
            qty=self.I[-1],
            normalize=True,
            price=True,
            size=True,
            levels = self.bookSize
        )
        state_next = ActionState(t_next, i_next, {'bidask': bidAskFeature})
        done = self.execution.isFilled() or state_next.getI() == 0
        # print(str((execution.getState().getT(), execution.getState().getI())) + " -> " + str((t_next, i_next)))
        # print(execution.getOrder().getCty())
        # print(execution.getQtyExecuted())
        # print(execution.getPcFilled())
        self.orderbookIndex = self.execution.getOrderbookIndex()
        self.actionState = state_next
        return state_next.toArray(), reward, done, {}

    def reset(self):
        return self._reset(t=self.T[-1], i=self.I[-1])

    def _reset(self, t, i):
        orderbookState, orderbookIndex = self._get_random_orderbook_state()
        bidAskFeature = self.orderbook.getBidAskFeatures(
            orderbookIndex,
            lookback=self.lookback,
            qty=self.I[-1],
            normalize=True,
            price=True,
            size=True,
            levels = self.bookSize
        )
        state = ActionState(t, i, {'bidask': bidAskFeature}) #np.array([[t, i]])
        self.execution = None
        self.orderbookIndex = orderbookIndex
        self.actionState = state
        return state.toArray()

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        pass


# import gym_ctc_executioner
# env = gym.make("ctc-executioner-v0")
# env.reset()
