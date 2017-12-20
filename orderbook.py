from dateutil import parser
from order_side import OrderSide
import numpy as np


class OrderbookEntry(object):

    def __init__(self, price, qty):
        self.price = price
        self.qty = qty

    def __str__(self):
        return str(self.price) + ": " + str(self.qty)

    def __repr__(self):
        return str(self)

    def getPrice(self):
        return self.price

    def getQty(self):
        return self.qty


class OrderbookState(object):

    def __init__(self, tradePrice=0.0, timestamp=None):
        self.tradePrice = tradePrice
        self.timestamp = timestamp
        self.buyers = []
        self.sellers = []

    def __str__(self):
        s = "DateTime: " + str(self.timestamp) + "\n"
        s = s + "Price: " + str(self.tradePrice) + "\n"
        s = s + "Buyers: " + str(self.buyers) + "\n"
        s = s + "Sellers: " + str(self.sellers)
        return s

    def __repr__(self):
        return str(self)

    def setTradePrice(self, tradePrice):
        self.tradePrice = tradePrice

    def addBuyer(self, entry):
        self.buyers.append(entry)

    def addBuyers(self, entries):
        for entry in entries:
            self.buyers.append(entry)

    def addSeller(self, entry):
        self.sellers.append(entry)

    def addSellers(self, entries):
        for entry in entries:
            self.sellers.append(entry)

    def getBuyers(self):
        return self.buyers

    def getSellers(self):
        return self.sellers

    def getTimestamp(self):
        return self.timestamp

    def getBidAskMid(self):
        firstBuy = self.getBuyers()[0]
        firstSell = self.getSellers()[0]
        return (firstBuy.getPrice() + firstSell.getPrice()) / 2.0

    def getSidePositions(self, side):
        if side == OrderSide.BUY:
            return self.getBuyers()
        elif side == OrderSide.SELL:
            return self.getSellers()

    def getBasePrice(self, side):
        return self.getSidePositions(side)[0].getPrice()

    def getPriceAtLevel(self, side, level):
        """ Returns price at a certain level of the orderbook.
        In case not enough levels are present in the orderbook, we assume
        linear price increase (seller side) / decrease (buyer side) for the
        following levels.
        """
        positions = self.getSidePositions(side)
        level = abs(level)
        if level < len(positions):
            return positions[level].getPrice()

        # Estimate subsequent levels
        derivative_price = np.mean(np.gradient([x.getPrice() for x in positions]))
        missingLevels = level - len(positions)
        priceEnd = positions[-1].getPrice()
        priceEstimated = priceEnd + missingLevels * derivative_price
        return priceEstimated

class Orderbook(object):

    def __init__(self):
        self.states = []

    def __str__(self):
        s = ''
        i = 1
        for state in self.states:
            s = s + 'State ' + str(i) + "\n"
            s = s + '-------' + "\n"
            s = s + str(state)
            s = s + "\n\n"
            i = i + 1
        return s

    def __repr__(self):
        return str(self)

    def addState(self, state):
        self.states.append(state)

    def addStates(self, states):
        for state in states:
            self.states.append(state)

    def getStates(self):
        return self.states

    def getState(self, index):
        if len(self.states) <= index:
            raise Exception('Index out of orderbook state.')
        return self.states[index]

    def getOffsetIndex(self, offset):
        """The index of the first state past the given offset in seconds.
        For example, if offset=3 with 10 states availble, whereas for
        simplicity every state has 1 second diff, then the resulting index
        would be the one marked with 'i':

        |x|x|x|i|_|_|_|_|_|_|

        As a result, the elements marked with 'x' are not meant to be used.

        """
        if offset == 0:
            return 0

        states = self.getStates()
        startState = states[0]
        offsetIndex = 0
        consumed = 0.0
        while(consumed < offset and offsetIndex < len(states)):
            state = states[offsetIndex]
            consumed = (state.getTimestamp() - startState.getTimestamp()).total_seconds()
            offsetIndex = offsetIndex + 1

        if consumed < offset:
            raise Exception('Not enough data for offset. Found states for '
                            + str(consumed) + ' seconds, required: '
                            + str(offset))

        return offsetIndex

    def getIndexWithTimeRemain(self, seconds, offset=0):
        """ Returns the state with seconds remaining starting from the end.
        For example, if seconds=3 and offset=1 with 10 states availble, whereas
        for simplicity every state has 1 second diff, then the resulting index
        would be the one marked with 'i' and the cap set by the offset is 'c'
        and 'x' indicating the non-usable elements:

        |x|c|_|_|_|_|i|>|>|>|

        As a result, the maximum seconds to retrieve would be 8, however only 3
        (4 counting the element of the index position) are used in this case
        and 4 are not being used (marked with '_').

        """
        if not self.getStates:
            raise Exception('Order book does not contain states.')

        states = self.getStates()
        endState = states[-1]
        index = len(states) - 2
        offsetIndex = self.getOffsetIndex(offset)
        consumed = 0.0
        while(consumed < seconds and index >= offsetIndex):
            state = states[index]
            consumed = (endState.getTimestamp() - state.getTimestamp()).total_seconds()
            index = index - 1

        if consumed < seconds:
            raise Exception('Not enough data available. Found states for '
                            + str(consumed) + ' seconds, required: '
                            + str(seconds))
        return index

    def getTotalDuration(self, offset=0):
        """Time span of data in seconds.
        The offset fixes the pointer starting from the beginning of the data
        feed.
        """
        states = self.getStates()
        offsetIndex = self.getOffsetIndex(offset)
        start = states[offsetIndex].getTimestamp()
        end = states[-1].getTimestamp()
        return (end - start).total_seconds()

    def loadFromFile(self, file):
        import csv
        with open(file, 'rt') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                p = float(row[1])
                b1 = float(row[3])
                b2 = float(row[4])
                b3 = float(row[5])
                b4 = float(row[6])
                b5 = float(row[7])
                a1 = float(row[8])
                a2 = float(row[9])
                a3 = float(row[10])
                a4 = float(row[11])
                a5 = float(row[12])
                bq1 = float(row[13])
                bq2 = float(row[14])
                bq3 = float(row[15])
                bq4 = float(row[16])
                bq5 = float(row[17])
                aq1 = float(row[18])
                aq2 = float(row[19])
                aq3 = float(row[20])
                aq4 = float(row[21])
                aq5 = float(row[22])
                dt = parser.parse(row[24])  # trade timestamp as reference
                buyers = [
                    OrderbookEntry(b1, bq1),
                    OrderbookEntry(b2, bq2),
                    OrderbookEntry(b3, bq3),
                    OrderbookEntry(b4, bq4),
                    OrderbookEntry(b5, bq5)
                ]
                sellers = [
                    OrderbookEntry(a1, aq1),
                    OrderbookEntry(a2, aq2),
                    OrderbookEntry(a3, aq3),
                    OrderbookEntry(a4, aq4),
                    OrderbookEntry(a5, aq5)
                ]
                s = OrderbookState(tradePrice=p, timestamp=dt)
                s.addBuyers(buyers)
                s.addSellers(sellers)
                self.addState(s)


# o = Orderbook()
# o.loadFromFile('query_result_small.tsv')
# print(o.getTotalDuration(offset=0))
# print(o.getIndexWithTimeRemain(seconds=99, offset=10))
# s0 = o.getState(0).getTimestamp()
# s1 = o.getState(1).getTimestamp()
# print(s0)
# print("")
# print(s1)
# print("")
# print((s1-s0).total_seconds())
