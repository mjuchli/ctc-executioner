import unittest
from action import Action
from orderbook import Orderbook
from order import Order
from order_type import OrderType
from order_side import OrderSide

orderbook = Orderbook(extraFeatures=False)
orderbook.loadFromFile('test_orderbook_10s.tsv')

class MatchEngineMock():

    def matchOrder(self, order, seconds=None):
        trades = []
        qtyRemain = 0
        index = 1
        return trades, qtyRemain, index


class ActionMock(Action):

    def __init__(self, a, runtime):
        Action.__init__(self, a, runtime)

    def getMatchEngine(self, orderbook):
        return MatchEngineMock()


class ActionTest(unittest.TestCase):

    def testRun(self):
        a = 1
        i = 1.0
        t = 10.0
        orderbookIndex = 0
        orderbookState = orderbook.getState(orderbookIndex)
        orderSide = OrderSide.BUY
        orderType = OrderType.LIMIT
        price = orderbookState.getPriceAtLevel(orderSide, a)
        order = Order(
            orderType=orderType,
            orderSide=orderSide,
            cty=i,
            price=price
        )
        action = ActionMock(a=a, runtime=t)
        action.setOrder(order)
        action.setOrderbookState(orderbookState)
        action.setOrderbookIndex(orderbookIndex)
        action.run(orderbook)
