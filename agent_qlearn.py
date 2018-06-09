import pickle
import logging
import numpy as np
from ctc_executioner.action_space_env import ActionSpace
from ctc_executioner.action_state import ActionState
from ctc_executioner.order_side import OrderSide
from ctc_executioner.qlearn import QLearn
from ctc_executioner.orderbook import Orderbook
from ctc_executioner.agent_utils.ui import UI

class AgentQlearn:
    def __init__(self, env):
        self.env = env
        self.levels = levels
        self.ai = QLearn(self.levels)
        self.orderbookIndex = None
        self.logRewards = []
        self.logActions = []

    def update(self, t, i, force_execution=False):
        aiState = ActionState(t, i)
        a = self.ai.chooseAction(aiState)
        self.logActions.append(a)
        # print('Random action: ' + str(level) + ' for state: ' + str(aiState))
        action = self.env.createAction(level=a, state=aiState, force_execution=force_execution, orderbookIndex=self.orderbookIndex)
        action.run(self.env.orderbook)
        i_next = self.env.determineNextInventory(action)
        t_next = self.env.determineNextTime(t)
        reward = action.getReward()
        self.logRewards.append(reward)
        state_next = ActionState(action.getState().getT(), action.getState().getI(), action.getState().getMarket())
        state_next.setT(t_next)
        state_next.setI(i_next)
        #print("Reward " + str(reward) + ": " + str(action.getState()) + " with " + str(action.getA()) + " -> " + str(state_next))
        self.ai.learn(
            state1=action.getState(),
            action1=action.getA(),
            reward=reward,
            state2=state_next
        )
        return (t_next, i_next)


    def train(self, episodes=1, force_execution=False):
        self.logRewards = []
        self.logActions = []
        for episode in range(int(episodes)):
            _, self.orderbookIndex = self.env.getRandomOrderbookState()
            for t in self.env.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.env.I:
                    self.orderbookIndex = self.orderbookIndex + 1
                    logging.info("     i=="+str(i))
                    logging.info("Action run " + str((t, i)))
                    (t_next, i_next) = self.update(t, i, force_execution)
                    while i_next != 0:
                        if force_execution:
                            raise Exception("Enforced execution left " + str(i_next) + " unexecuted.")
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))
                        (t_next, i_next) = self.update(t_next, i_next, force_execution)

    def backtest(self, q=None, episodes=10, average=False, fixed_a=None):
        Ms = []
        for _ in range(episodes):
            actions = []
            t = self.env.T[-1]
            i = self.env.I[-1]
            state = ActionState(t, i, {})
            #print(state)
            if fixed_a is not None:
                a = fixed_a
            else:
                a = self.ai.getQAction(state, 0)

            actions.append(a)
            action = self.env.createAction(level=a, state=state, force_execution=True)
            midPrice = action.getReferencePrice()

            #print("before...")
            #print(action)
            action.run(self.env.orderbook)
            #print("after...")
            #print(action)
            i_next = self.env.determineNextInventory(action)
            t_next = self.env.determineNextTime(t)
            # print("i_next: " + str(i_next))
            while i_next != 0:
                state_next = ActionState(t_next, i_next, {})
                if fixed_a is not None:
                    a_next = fixed_a
                else:
                    a_next = self.ai.getQAction(state_next, 0)

                actions.append(a_next)
                #print("Action transition " + str((t, i)) + " -> " + str(aiState_next) + " with " + str(runtime_next) + "s runtime.")

                runtime_next = self.env.determineRuntime(t_next)
                action.setState(state_next)
                action.update(a_next, runtime_next)
                action.run(self.env.orderbook)
                #print(action)
                i_next = self.env.determineNextInventory(action)
                t_next = self.env.determineNextTime(t_next)

            price = action.getAvgPrice()
            if action.getOrder().getSide() == OrderSide.BUY:
                profit = midPrice - price
            else:
                profit = price - midPrice
            Ms.append([state, midPrice, actions, price, profit])
        if not average:
            return Ms
        return self.averageBacktest(Ms)

    def averageBacktest(self, M):
        # Average states within M
        N = []
        observed = []
        for x in M:
            state = x[0]
            if state in observed:
                continue
            observed.append(state)
            paid = []
            reward = []
            for y in M:
                if y[0] == state:
                    paid.append(y[3])
                    reward.append(y[4])
            N.append([state, x[1], x[2], np.average(paid), np.average(reward)])
        return N

    def run(self, epochs_train=1, epochs_test=10):
        if epochs_train > 0:
            agent.train(episodes=epochs_train)
            rewards = agent.logRewards
            actions = agent.logActions
            #print(actions)
            return np.mean(rewards)

        if epochs_test > 0:
            M = agent.backtest(episodes=epochs_test, average=False)
            M = np.array(M)
            return np.mean(M[0:, 4])

    def simulate(self, epochs_train=1, epochs_test=10, interval=100):
        UI.animate(lambda : self.run(epochs_train, epochs_test), interval=interval, title="Mean backtest reward")


def _generate_Sequence(min, max, step):
    """ Generate sequence (that unlike xrange supports float)

    max: defines the sequence maximum
    step: defines the interval
    """
    i = min
    I = []
    while i <= max:
        I.append(i)
        i = i + step
    return I


side = OrderSide.SELL
dataset = "2"
name = "experiments/q_"+dataset+"_10000_" + str(side)
levels = _generate_Sequence(min=-50, max=50, step=1)
ai = None
T = _generate_Sequence(min=0, max=100, step=10)
T_test = _generate_Sequence(min=0, max=100, step=10)
I = _generate_Sequence(min=0, max=1, step=0.1)

# Load orderbook
cols = ["ts", "seq", "size", "price", "is_bid", "is_trade", "ttype"]
import pandas as pd
events = pd.read_table('data/events/ob-'+dataset+'-small-train.tsv', sep='\t', names=cols, index_col="seq")
d = Orderbook.generateDictFromEvents(events)
orderbook = Orderbook()
orderbook.loadFromDict(d)
# clean first n states (due to lack of bids and asks)
print("#States: " + str(len(orderbook.states)))


events_test = pd.read_table('data/events/ob-'+dataset+'-small-test.tsv', sep='\t', names=cols, index_col="seq")
d_test = Orderbook.generateDictFromEvents(events_test)
orderbook_test = Orderbook()
orderbook_test.loadFromDict(d_test)

for i in range(25):
    orderbook.states.pop(0)
    orderbook_test.states.pop(0)
    del d[list(d.keys())[0]]
    del d_test[list(d_test.keys())[0]]

#orderbook.plot()
#orderbook_test.plot()

actionSpace = ActionSpace(orderbook, side, T, I, levels=levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, levels=levels)
agent = AgentQlearn(actionSpace)

# TRAIN
# actions = []
# rewards = []
# print("Learn " + name)
# for i in range(6000):
#     print("Epoch: " + str(i))
#     try:
#         agent.train(episodes=1)
#         actions.append(agent.logActions)
#         rewards.append(agent.logRewards)
#     except:
#         print("Index error")
#
# np.save(name+'.npy', agent.ai.q)
#
# with open(name + '_actions', 'wb') as fp:
#     pickle.dump(actions, fp)
# with open(name  + '_rewards', 'wb') as fp:
#     pickle.dump(rewards, fp)


#agent.simulate(epochs_train=1, epochs_test=0)

# TEST
agent_test = AgentQlearn(actionSpace_test)
q = np.load(name+'.npy').item()
agent_test.ai.q = q

print("Test " + name)
backtest = []
for i in range(1000):
    print("Test: " + str(i))
    #try:
    M = agent.backtest(episodes=1, average=False, fixed_a=0)
    M = np.array(M)
    reward = np.mean(M[0:, 4])
    #print(reward)
    backtest.append(reward)
    #except:
    #    print("Index error")

print(np.mean(backtest))
with open(name  + '_backtest', 'wb') as fp:
    pickle.dump(backtest, fp)

#agent_test.simulate(epochs_train=0, epochs_test=10)
