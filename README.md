# Order execution with Reinforcement Learning 

CTC-Executioner is a tool that provides an on-demand execution strategy for limit orders on crypto currency markets using Reinforcement Learning techniques. The underlying framework provides functionalities which allow to analyse order book data and derive features thereof. Those findings can then be used in order to dynamically update the decision making process of the execution strategy.

The methods being used are based on a research project (master thesis) currently proceeding at TU Delft.

## Documentation

See [Wiki](https://github.com/backender/ctc-executioner/wiki)

## Usage

Load orderbooks

```python
orderbook = Orderbook()
orderbook.loadFromEvents('data/example-ob-train.tsv')
orderbook.summary()
orderbook.plot(show_bidask=True)

orderbook_test = Orderbook()
orderbook_test.loadFromEvents('data/example-ob-test.tsv')
orderbook_test.summary()
```

Create and configure environments

```python
import gym_ctc_executioner
env = gym.make("ctc-executioner-v0")
env.setOrderbook(orderbook)

env_test = gym.make("ctc-executioner-v0")
env_test.setOrderbook(orderbook_test)
```
