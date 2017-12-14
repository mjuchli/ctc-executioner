from trade import Trade
from order import Order
from order_type import OrderType
from order_side import OrderSide
import copy
import logging


class MatchEngine(object):

    def __init__(self, orderbook, index=0):
        self.orderbook = orderbook
        self.index = index

    def setIndex(self, index):
        self.index = index

    def matchLimitOrder(self, order, orderbookState):
        if order.getSide() == OrderSide.BUY:
            bookSide = orderbookState.getSellers()
        else:
            bookSide = orderbookState.getBuyers()

        partialTrades = []
        remaining = order.getCty()
        sidePosition = 0
        while len(bookSide) > sidePosition and bookSide[sidePosition].getPrice() <= order.getPrice() and remaining > 0.0:
            p = bookSide[sidePosition]
            price = p.getPrice()
            qty = p.getQty()
            if not partialTrades and qty >= order.getCty():
                logging.info("Full execution: " + str(qty) + " pcs available")
                return [Trade(orderSide=order.getSide(), cty=remaining, price=price)]
            else:
                logging.info("Partial execution: " + str(qty) + " pcs available")
                partialTrades.append(Trade(orderSide=order.getSide(), cty=min(qty, remaining), price=price))
                sidePosition = sidePosition + 1
                remaining = remaining - qty
        return partialTrades

    def matchMarketOrder(self, order, orderbookState):
        if order.getSide() == OrderSide.BUY:
            bookSide = orderbookState.getSellers()
        else:
            bookSide = orderbookState.getBuyers()

        partialTrades = []
        remaining = order.getCty()
        sidePosition = 0
        price = 0.0
        while len(bookSide) > sidePosition and remaining > 0.0:
            p = bookSide[sidePosition]
            derivative_price = p.getPrice() - price
            price = p.getPrice()
            qty = p.getQty()
            if not partialTrades and qty >= order.getCty():
                logging.info("Full execution: " + str(qty) + " pcs available")
                return [Trade(orderSide=order.getSide(), cty=remaining, price=price)]
            else:
                logging.info("Partial execution: " + str(qty) + " pcs available")
                partialTrades.append(Trade(orderSide=order.getSide(), cty=min(qty, remaining), price=price))
                sidePosition = sidePosition + 1
                remaining = remaining - qty

        # Since there there is no more liquidity in this state of the order
        # book (data). For convenience sake we assume that there would be
        # liquidity in some levels beyond.
        # TODO: Simulate in more appropriate way, such as executing multiple
        # trades whereas the trade size increases exponentially and the price
        # increases logarithmically.
        if remaining > 0.0:
            price = price + derivative_price
            logging.info("Partial execution: assume " + str(remaining) + " availabile")
            partialTrades.append(Trade(orderSide=order.getSide(), cty=remaining, price=price))

        return partialTrades

    def matchOrder(self, order, seconds=None):
        order = copy.deepcopy(order)  # Do not modify original order!
        i = self.index
        remaining = order.getCty()
        trades = []
        while len(self.orderbook.getStates()) > i and remaining > 0:
            orderbookState = self.orderbook.getState(i)
            logging.info("Evaluate state " + str(i) + ":\n" + str(orderbookState))

            # Stop matching process after defined seconds are consumed
            if seconds:
                t_start = self.orderbook.getState(self.index).getTimestamp()
                t_now = orderbookState.getTimestamp()
                t_delta = (t_now - t_start).total_seconds()
                logging.info(str(t_delta) + " of " + str(seconds) + " consumed.")
                if t_delta >= seconds:
                    logging.info("Time delta consumed, stop matching.")
                    break

            if order.getType() == OrderType.LIMIT:
                counterTrades = self.matchLimitOrder(order, orderbookState)
            elif order.getType() == OrderType.MARKET:
                counterTrades = self.matchMarketOrder(order, orderbookState)
            else:
                raise Exception('Order type not known or not implemented yet.')

            if counterTrades:
                trades = trades + counterTrades
                logging.info("Trades executed:")
                for counterTrade in counterTrades:
                    logging.info(counterTrade)
                    remaining = remaining - counterTrade.getCty()
                order.setCty(remaining)
                logging.info("Remaining: " + str(remaining) + "\n")
            else:
                logging.info("No orders matched.\n")
            i = i + 1
        logging.info("Total number of trades: " + str(len(trades)))
        logging.info("Remaining qty of order: " + str(remaining))
        return trades, remaining


# logging.basicConfig(level=logging.INFO)
# from orderbook import Orderbook
# orderbook = Orderbook()
# orderbook.loadFromFile('query_result_small.tsv')
# engine = MatchEngine(orderbook, index=0)
#
# order = Order(orderType=OrderType.LIMIT, orderSide=OrderSide.BUY, cty=11.0, price=16559.0)
# #order = Order(orderType=OrderType.MARKET, orderSide=OrderSide.BUY, cty=25.5, price=None)
# trades, remaining = engine.matchOrder(order, seconds=1.0)
# c = 0.0
# for trade in trades:
#     c = c + trade.getCty()
# print(c)
