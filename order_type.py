from enum import Enum


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    CANCEL = 'cancel'
