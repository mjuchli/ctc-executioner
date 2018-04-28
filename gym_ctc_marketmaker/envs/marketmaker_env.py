import logging
import itertools
from gym import spaces
from ctc_executioner.action import Action
from ctc_executioner.action_state import ActionState
from ctc_executioner.order import Order
from ctc_executioner.order_type import OrderType
from ctc_executioner.order_side import OrderSide
import gym_ctc_executioner.envs.execution_env as execution_env
#logging.basicConfig(level=logging.INFO)

class MarketMakerEnv(execution_env.ExecutionEnv):

    def __init__(self):
        self.orderbookIndexBuy = None
        self.orderbookIndexSell = None
        self.actionStateBuy = None
        self.actionStateSell = None
        self.executionBuy = None
        self.executionSell = None
        self._configure()

    def _configure(self,
                   orderbook=None,
                   levels=(-50, 50, 1),
                   T=(0, 100, 10),
                   I=(0, 1, 0.1),
                   lookback=25,
                   bookSize=10
                   ):
        self.orderbook = orderbook
        self.levels = self._generate_Sequence(min=levels[0], max=levels[1], step=levels[2])
        self.levels = list(itertools.product(self.levels, self.levels))
        self.T = self._generate_Sequence(min=T[0], max=T[1], step=T[2])
        self.I = self._generate_Sequence(min=I[0], max=I[1], step=I[2])
        self.lookback = lookback # results in (bid|size, ask|size) -> 4*5
        self.bookSize = bookSize
        self.action_space = spaces.Discrete(len(self.levels))
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2*self.lookback, self.bookSize, 2))

    def setSide(self, side):
        pass

    def _create_execution(self, a, actionState, orderbookIndex, side):
        runtime = self._determine_runtime(actionState.getT())
        orderbookState = self.orderbook.getState(orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = orderbookState.getPriceAtLevel(side, a)
            ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=side,
            cty=actionState.getI(),
            price=price
        )
        execution = Action(a=a, runtime=runtime)
        execution.setState(actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(orderbookIndex)
        execution.setReferencePrice(orderbookState.getBestAsk())
        return execution

    def _update_execution(self, execution, a, actionState, orderbookIndex, side):
        runtime = self._determine_runtime(actionState.getT())
        orderbookState = self.orderbook.getState(orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = execution.getOrderbookState().getPriceAtLevel(side, a)
            ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=side,
            cty=actionState.getI(),
            price=price
        )
        execution.setRuntime(runtime)
        execution.setState(actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(orderbookIndex)
        return execution

    def step(self, action):
        # print('action')
        # print(action)
        actionBuy = self.levels[action][0]
        actionSell = self.levels[action][1]

        if self.executionBuy is None or self.executionSell is None:
            self.executionBuy = self._create_execution(a=actionBuy, actionState=self.actionStateBuy, orderbookIndex=self.orderbookIndexBuy, side=OrderSide.BUY)
            self.executionSell = self._create_execution(a=actionSell, actionState=self.actionStateSell, orderbookIndex=self.orderbookIndexSell, side=OrderSide.SELL)
        else:
            if not self.executionBuy.isFilled():
                self.executionBuy = self._update_execution(execution=self.executionBuy, a=actionBuy, actionState=self.actionStateBuy, orderbookIndex=self.orderbookIndexBuy, side=OrderSide.BUY)
            if not self.executionSell.isFilled():
                self.executionSell = self._update_execution(execution=self.executionSell, a=actionSell, actionState=self.actionStateSell, orderbookIndex=self.orderbookIndexSell, side=OrderSide.SELL)

        # logging.info(
        #     'Created/Updated execution.' +
        #     '\nAction: ' + str(action) + ' (' + str(self.execution.getOrder().getType()) + ')' +
        #     '\nt: ' + str(self.actionState.getT()) +
        #     '\nruntime: ' + str(self.execution.getRuntime()) +
        #     '\ni: ' + str(self.actionState.getI())
        # )
        if not self.executionBuy.isFilled():
            self.executionBuy, counterTradesBuy = self.executionBuy.run(self.orderbook)
            i_next_buy = self._determine_next_inventory(self.executionBuy)
            t_next_buy = self._determine_next_time(self.executionBuy.getState().getT())
            bidAskFeatureBuy = self._makeFeature(orderbookIndex=self.executionBuy.getOrderbookIndex())
            self.actionStateBuy = ActionState(t_next_buy, i_next_buy, {'bidask': bidAskFeatureBuy})
            self.orderbookIndexBuy = self.executionBuy.getOrderbookIndex()
            price_buy = self.executionBuy.calculateAvgPrice(counterTradesBuy)
        else:
            price_buy = self.executionBuy.getAvgPrice()

        if not self.executionSell.isFilled():
            self.executionSell, counterTradesSell = self.executionSell.run(self.orderbook)
            i_next_sell = self._determine_next_inventory(self.executionSell)
            t_next_sell = self._determine_next_time(self.executionSell.getState().getT())
            bidAskFeatureSell = self._makeFeature(orderbookIndex=self.executionSell.getOrderbookIndex())
            self.actionStateSell = ActionState(t_next_sell, i_next_sell, {'bidask': bidAskFeatureSell})
            self.orderbookIndexSell = self.executionSell.getOrderbookIndex()
            price_sell = self.executionSell.calculateAvgPrice(counterTradesSell)
        else:
            price_sell = self.executionSell.getAvgPrice()


        done_buy = self.executionBuy.isFilled() or self.actionStateBuy.getI() == 0
        done_sell = self.executionSell.isFilled() or self.actionStateSell.getI() == 0

        print('price buy: ' + str(price_buy))
        print('price sell: ' + str(price_sell))
        if price_buy == 0 or price_sell == 0:
            reward = 0.0
        else:
            reward = price_sell - price_buy
        print('reward: ' + str(reward))

        # logging.info(
        #     'Run execution.' +
        #     '\nTrades: ' + str(len(counterTrades)) +
        #     '\nReward: ' + str(reward) + ' (Ratio: ' + str(volumeRatio) + ')' +
        #     '\nDone: ' + str(done)
        # )

        if self.orderbookIndexBuy >= self.orderbookIndexSell:
            state_next = self.actionStateBuy
        else:
            state_next = self.actionStateSell
        return state_next.toArray(), reward, (done_buy and done_sell), {}

    def reset(self):
        return self._reset(t=self.T[-1], i=self.I[-1])

    def _reset(self, t, i):
        #self.orderbook = copy.deepcopy(self.orderbookOriginal) # TODO: Slow but currently required to reset after every episode due to change of order book states during matching
        orderbookState, orderbookIndex = self._get_random_orderbook_state()
        bidAskFeature = self._makeFeature(orderbookIndex=orderbookIndex)
        state = ActionState(t, i, {'bidask': bidAskFeature}) #np.array([[t, i]])

        self.executionBuy = None
        self.executionSell = None

        self.orderbookIndexBuy = orderbookIndex
        self.orderbookIndexSell = orderbookIndex

        self.actionStateBuy = state
        self.actionStateSell = state

        return state.toArray()

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        pass
