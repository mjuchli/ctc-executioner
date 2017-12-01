from datetime import datetime

""" A trade is an extended version of a Position, indicating the purchases made """
class Trade:
    def __init__(self, orderType, cty, price, fee, timestamp = str(datetime.now()).split('.')[0]):
        self.orderType = orderType
        self.cty = cty
        self.price = price
        self.fee = fee
        self.timestamp = timestamp

    def __str__(self):
        return (str(self.timestamp) + ',' +
                str(self.getType()) + ',' +
                str(self.getCty()) + ',' +
                str(self.getPrice()) + ',' +
                str(self.getFee()))

    def getType(self):
        return self.orderType

    def getCty(self):
        return self.cty

    def setCty(self, cty):
        self.cty = cty

    def getPrice(self):
        return self.price

    def getFee(self):
        return self.fee

    def getTimeStamp(self):
        return self.timestamp
