import unittest
from action import Action
from orderbook import Orderbook


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

    def getOrderbook(self):
        o = Orderbook()
        o.addState

    def testRun(self):
        orderbook = self.getOrderbook()
        action = ActionMock(a=1, runtime=1)
        action, qtyRemain = action.run(orderbook)
        print(qtyRemain)


at = ActionTest()
at.testRun()
