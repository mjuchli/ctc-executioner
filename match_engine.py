from trade import Trade
from order_type import OrderType
import copy
import logging

class MatchEngine(object):

    def __init__(self, orderbook, index=0):
        self.orderbook = orderbook
        self.index = index

    def setIndex(self, index):
        self.index = index

    def matchLimitOrder(self, order, orderbookState):
        if order.getType() == OrderType.BUY:
            bookSide = orderbookState.getSellers()
        else:
            bookSide = orderbookState.getBuyers()

        partialTrades = []
        totalQty = order.getCty()
        sidePosition = 0
        while len(bookSide) > sidePosition and bookSide[sidePosition].getPrice() <= order.getPrice():
            p = bookSide[sidePosition]
            price = p.getPrice()
            qty = p.getQty()
            if qty >= totalQty:
                logging.info("Full execution: " + str(qty) + " pcs available")
                return [Trade(order.getType(), totalQty, price)]
            else:
                logging.info("Partial execution: " + str(qty) + " pcs available")
                partialTrades.append(Trade(order.getType(), qty, price))
                sidePosition = sidePosition + 1
        return partialTrades

    def matchOrderOverTime(self, order):
        order = copy.deepcopy(order)  # Do not modify original order!
        i = self.index
        remaining = order.getCty()
        trades = []
        while len(self.orderbook.getStates()) > i and remaining > 0:
            orderbookState = self.orderbook.getState(i)
            counterTrades = self.matchLimitOrder(order, orderbookState)
            if counterTrades:
                trades = trades + counterTrades
                logging.info("Trades executed:")
                for counterTrade in counterTrades:
                    logging.info(counterTrade)
                    remaining = remaining - counterTrade.getCty()
                order.setCty(remaining)
                logging.info("In state " + str(i) + ":\n" + str(orderbookState))
                logging.info("Remaining: " + str(remaining) + "\n")
            i = i + 1
        logging.info("Total number of trades: " + str(len(trades)))
        return trades, remaining


# logging.basicConfig(level=logging.INFO)
# from orderbook import Orderbook
# orderbook = Orderbook()
# orderbook.loadFromFile('query_result_small.tsv')
# engine = MatchEngine(orderbook, index=0)
#
# order = Trade(orderType=OrderType.BUY, cty=20.0, price=16559.0)
# engine.matchOrderOverTime(order)
