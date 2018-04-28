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

    def update(self, t, i, force_execution=False):
        aiState = ActionState(t, i)
        a = self.ai.chooseAction(aiState)
        # print('Random action: ' + str(level) + ' for state: ' + str(aiState))
        action = self.env.createAction(level=a, state=aiState, force_execution=force_execution)
        action.run(self.env.orderbook)
        i_next = self.env.determineNextInventory(action)
        t_next = self.env.determineNextTime(t)
        reward = action.getReward()
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
        for episode in range(int(episodes)):
            for t in self.env.T:
                logging.info("\n"+"t=="+str(t))
                for i in self.env.I:
                    logging.info("     i=="+str(i))
                    logging.info("Action run " + str((t, i)))
                    (t_next, i_next) = self.update(t, i, force_execution)
                    while i_next != 0:
                        if force_execution:
                            raise Exception("Enforced execution left " + str(i_next) + " unexecuted.")
                        logging.info("Action transition " + str((t, i)) + " -> " + str((t_next, i_next)))
                        (t_next, i_next) = self.update(t_next, i_next, force_execution)


    def backtest(self, q=None, episodes=10, average=False, fixed_a=None):
        if q is None:
            q = self.ai.q
        else:
            self.ai.q = q

        if not q:
            raise Exception('Q-Table is empty, please train first.')

        Ms = []
        #T = self.T[1:len(self.T)]
        for t in [self.env.T[-1]]:
            logging.info("\n"+"t=="+str(t))
            for i in [self.env.I[-1]]:
                logging.info("     i=="+str(i))
                actions = []
                state = ActionState(t, i, {})
                #print(state)
                if fixed_a is not None:
                    a = fixed_a
                else:
                    try:
                        a = self.ai.getQAction(state, 0)
                        print("t: " + str(t))
                        print("i: " + str(i))
                        print("Action: " + str(a))
                        # print("Q action for state " + str(state) + ": " + str(a))
                    except:
                        # State might not be in Q-Table yet, more training requried.
                        logging.info("State " + str(state) + " not in Q-Table.")
                        break
                actions.append(a)
                action = self.env.createAction(level=a, state=state, force_execution=False)
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
                        try:
                            a_next = self.ai.getQAction(state_next, 0)
                            print("t: " + str(t_next))
                            print("i: " + str(i_next))
                            print("Action: " + str(a_next))
                            # print("Q action for next state " + str(state_next) + ": " + str(a_next))
                        except:
                            # State might not be in Q-Table yet, more training requried.
                            # print("State " + str(state_next) + " not in Q-Table.")
                            break
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
                # TODO: last column is for for the BUY scenario only
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
        M = agent.backtest(episodes=epochs_test, average=False)
        M = np.array(M)
        return np.mean(M[0:, 4])

    def simulate(self, epochs_train=1, epochs_test=10, interval=100):
        UI.animate(lambda : self.run(epochs_train, epochs_test), interval=interval)


side = OrderSide.SELL
levels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -9]
ai = None
T = [0, 10, 20, 40, 60, 80, 100] #, 120, 240]
T_test = [0, 10, 20, 40, 60, 80, 100]# 120, 240]
I = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Load orderbook
cols = ["ts", "seq", "size", "price", "is_bid", "is_trade", "ttype"]
import pandas as pd
events = pd.read_table('data/events/ob-1-small.tsv', sep='\t', names=cols, index_col="seq")
d = Orderbook.generateDictFromEvents(events)
orderbook = Orderbook()
orderbook.loadFromDict(d)
# clean first n states (due to lack of bids and asks)
print("#States: " + str(len(orderbook.states)))
for i in range(100):
    orderbook.states.pop(0)
    del d[list(d.keys())[0]]
orderbook_test = orderbook
#orderbook.plot()

actionSpace = ActionSpace(orderbook, side, T, I, levels=levels)
actionSpace_test = ActionSpace(orderbook_test, side, T_test, I, levels=levels)
agent = AgentQlearn(actionSpace)
agent.simulate()
