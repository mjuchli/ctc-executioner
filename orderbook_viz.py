"""Orderbook visualization and testing utilities.

This module contains test code and visualization utilities for the Orderbook class.
Moved from ctc_executioner/orderbook.py for cleaner separation.
"""

import datetime
from ctc_executioner.orderbook import Orderbook
import numpy as np


def main():
    """Main function for orderbook visualization and testing."""
    # Example 1: Create artificial orderbook and plot
    # orderbook = Orderbook()
    # config = {
    #     "startPrice": 10010.0,
    #     "priceFunction": lambda p0, s, samples: p0
    #     + 10 * np.sin(2 * np.pi * 5 * (s / samples)),  # 10*sine with interval=5
    #     "levels": 25,
    #     "qtyPosition": 1.0,
    #     "startTime": datetime.datetime.now(),
    #     "duration": datetime.timedelta(minutes=10),
    #     "interval": datetime.timedelta(seconds=10),
    # }
    # orderbook.createArtificial(config)
    # orderbook.plot()
    # print("states: " + str(len(orderbook.getStates())))
    # st, index = orderbook.getRandomState(runtime=60, min_head=10)
    # print(f"Random state index: {index}")

    # Example 2: Load from events file
    o = Orderbook()
    o.loadFromEvents("data/events/ob-train.tsv")
    o.plot()
    ts = o.getStates()[100].getUnixTimestamp()
    print(ts)
    trades = o.getHistTradesFeature(ts, normalize=True, norm_price=2.0, norm_size=2.0)
    # trades = o.getHistTradesFeature(ts)
    print(trades[0:1])

    # Example 3: Load from file
    # o = Orderbook()
    # o.loadFromFile("query_result_small.tsv")

    # Example 3: Load from events
    # o = Orderbook()
    # o.loadFromEvents("data/events/ob-1-small.tsv")
    # ts = o.getStates()[100].getUnixTimestamp()
    # print(ts)
    # o.getHistTradesFeature(ts, normalize=True, norm_price=2.0, norm_size=2.0)
    # trades = o.getHistTradesFeature(ts)
    # print(trades[0:1])

    # Example 4: Load from Bitfinex file
    # o = Orderbook()
    # o.loadFromBitfinexFile("../ctc-executioner/orderbook_bitfinex_btcusd_view.tsv")
    # o.loadFromFile("query_result_train_15m.tsv")
    # o.plot()
    # o.createFeatures()
    # print([x.getMarketVar("volumeRelativeTotal") for x in o.getStates()])
    # print(o.getState(0))
    # print(o.getState(200))

    # Example 5: Test state access
    # print(o.getState(0))
    # print(o.getState(1))
    # print(o.getState(2))

    # Example 6: Test offset calculations
    # print(o.getTotalDuration(offset=0))
    # print(o.getOffsetTail(offset=0))
    # print(o.getOffsetTail(offset=16))
    # print(o.getRandomOffset(offset_max=60))
    # print(o.getIndexWithTimeRemain(seconds=60, offset=50))
    # s0 = o.getState(0).getTimestamp()
    # s1 = o.getState(1).getTimestamp()
    # print(s0)
    # print("")
    # print(s1)
    # print("")
    # print((s1 - s0).total_seconds())


if __name__ == "__main__":
    main()
