import copy
from order_side import OrderSide


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

    def getValueAbs(self, midPrice):
        """Retuns difference of the paid amount to the total bid/ask-mid amount.
        The higher, the better,
        For BUY: total paid at mid price - total paid
        For SELL: total received - total received at mid price
        """
        # In case of no executed trade, the value is the negative reference
        if self.getTotalPaidReceived() == 0.0:
            return -midPrice

        if self.getOrder().getSide() == OrderSide.BUY:
            return midPrice - self.getTotalPaidReceived()
        else:
            return self.getTotalPaidReceived() - midPrice

    def getValueAvg(self, midPrice):
        """Retuns difference of the average paid price to bid/ask-mid price.
        The higher, the better,
        For BUY: total paid at mid price - total paid
        For SELL: total received - total received at mid price
        """
        # In case of no executed trade, the value is the negative reference
        if self.getAvgPrice() == 0.0:
            return -2.0 * abs(self.getA())

        if self.getOrder().getSide() == OrderSide.BUY:
            return (1.0 - (midPrice / self.getAvgPrice()))
        else:
            return (1.0 - (self.getAvgPrice() / midPrice))

    def move(self, t_next, i_next):
        newAction = copy.deepcopy(self)
        newAction.setRuntime(t_next)
        newAction.getOrder().setCty(i_next)
        return newAction
