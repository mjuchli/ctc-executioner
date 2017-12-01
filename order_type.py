from enum import Enum


class OrderType(Enum):
    BUY = 'buy'
    SELL = 'sell'

    def opposite(self):
        if self == OrderType.BUY:
            return OrderType.SELL
        elif self == OrderType.SELL:
            return OrderType.BUY
