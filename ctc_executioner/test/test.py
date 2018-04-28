from order_side import OrderSide
from orderbook import Orderbook

orderbook = Orderbook()
orderbook.loadFromEvents('ob-1-small.tsv')
orderbook_test = orderbook
orderbook.summary()

side = OrderSide.SELL
levels = list(range(-20,21))

episode = {'episode': 9, 'steps': {
    0: {'action': 2, 'index': 167, 't': 100, 'i': 0.9999999999999999, 'reward': -6.224232140000822},
    1: {'action': 39, 'index': 173, 't': 90, 'i': 0.9999999999999999, 'reward': 1.9899999999997817},
    2: {'action': 21, 'index': 179, 't': 80, 'i': 0.7, 'reward': -6.224232140000822},
    3: {'action': 39, 'index': 185, 't': 70, 'i': 0.7, 'reward': -0.18051749500045844},
    4: {'action': 39, 'index': 193, 't': 60, 'i': 0.7, 'reward': -0.3610349900009169}
    }
 }

import matplotlib.pyplot as plt

indices = []
times = []
actions = []
prices = []
order_prices = []
runtimes = []
inventories = []
rewards = []
for key, value in episode['steps'].items():
    index = value['index']
    indices.append(index)
    runtimes.append(value['t'])
    inventories.append(value['i'])
    rewards.append(value['reward'])
    actions.append(value['action'])
    state = orderbook.getState(index)
    prices.append(state.getBidAskMid())
    action_delta = 0.1*levels[value['action']]
    if side == OrderSide.BUY:
        order_prices.append(state.getBidAskMid() + action_delta)
    else:
        order_prices.append(state.getBidAskMid() - action_delta)
    times.append(state.getTimestamp())

# price chart
ps = [x.getBidAskMid() for x in orderbook.getStates()]
ts = [x.getTimestamp() for x in orderbook.getStates()]
plt.plot(ts, ps)
# if show_bidask:
max_level = 0
buyer = [x.getBuyers()[max_level].getPrice() for x in orderbook.getStates()]
seller = [x.getSellers()[max_level].getPrice() for x in orderbook.getStates()]
plt.plot(ts, buyer)
plt.plot(ts, seller)

# order placement
#plt.scatter(times, prices, s=60)
plt.scatter(times, order_prices, s=60)

for i, time in enumerate(times):
    if i == 0 or i == len(times)-1:
        style = 'k-'
    else:
        style = 'k--'
    # line at order placement
    plt.plot([time, time], [prices[i]-0.005*prices[i], prices[i]+0.005*prices[i]], style, lw=1)


for i, action in enumerate(actions):
    # action, resulted reward
    txt = 'a='+str(levels[action]) + '\nr=' + str(round(rewards[i], 2))
    plt.annotate(txt, (times[i],prices[i]))
    # runtime, inventory
    txt = 't=' + str(runtimes[i]) + '\ni='+ str(round(inventories[i], 2))
    plt.annotate(txt, (times[i], prices[i]-0.005*prices[i]))

plt.show()
