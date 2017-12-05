from trade import Trade
from order_type import OrderType
import copy

class MatchEngine(object):

    def __init__(self, orderbook, index=0):
        self.orderbook = orderbook
        self.index = index

    def setIndex(self, index):
        self.index = index

    def matchTrade(self, trade, orderbookState):
        sellers = orderbookState.getSellers()
        buyers = orderbookState.getBuyers()
        totalQty = trade.getCty()
        if trade.getType == OrderType.BUY:
            for p in sellers:
                price = p.getPrice()
                qty = p.getQty()
                if price == trade.getPrice():
                    if qty >= totalQty:
                        return Trade(OrderType.SELL, totalQty, price)
                    else:
                        # partial execution
                        return Trade(OrderType.SELL, qty, price)
        else:
            for p in buyers:
                price = p.getPrice()
                qty = p.getQty()
                if price == trade.getPrice():
                    if qty >= totalQty:
                        return Trade(OrderType.BUY, totalQty, price, 0.0)
                    else:
                        # partial execution
                        return Trade(OrderType.BUY, qty, price, 0.0)

    def matchTradeOverTime(self, trade):
        trade = copy.deepcopy(trade)  # Do not modify original trade!
        i = self.index
        remaining = trade.getCty()
        trades = []
        while len(self.orderbook.getStates()) > i and remaining > 0:
            orderbookState = self.orderbook.getState(i)
            counterTrade = self.matchTrade(trade, orderbookState)
            if counterTrade:
                trades.append(counterTrade)
                print("counter trade: " + str(counterTrade))
                remaining = remaining - counterTrade.getCty()
                trade.setCty(remaining)
            i = i+1
        return trades, remaining
