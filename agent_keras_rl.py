from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import logging
import numpy as np
from order_side import OrderSide
from orderbook import Orderbook
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, SGD
from keras import regularizers

from agent_utils.action_plot_callback import ActionPlotCallback
from agent_utils.live_plot_callback import LivePlotCallback

from keras import optimizers
from collections import deque
import gym

#logging.basicConfig(level=logging.INFO)

from rl.callbacks import Callback
class EpsDecayCallback(Callback):
    def __init__(self, eps_poilcy, decay_rate=0.95):
        self.eps_poilcy = eps_poilcy
        self.decay_rate = decay_rate
    def on_episode_begin(self, episode, logs={}):
        self.eps_poilcy.eps *= self.decay_rate
        print('eps = %s' % self.eps_poilcy.eps)

def createModel():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Flatten(input_shape=(1,1,)+env.observation_space.shape))
    #model.add(Dense(4*env.bookSize*env.lookback))
    #model.add(Dense(env.bookSize*env.lookback))#, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(4*env.bookSize))
    model.add(Activation('relu'))
    model.add(Dense(len(env.levels)))
    model.add(Activation('linear'))
    #model.compile(optimizers.SGD(lr=.1), "mae")
    model.summary()
    return model

def loadModel(name):
    # load json and create model
    from keras.models import model_from_json
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(name + '.h5')
    print('Loaded model "' + name + '" from disk')
    return model

def saveModel(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + '.h5')
    print('Saved model "' + name + '" to disk')



# Load orderbook
orderbook = Orderbook()
orderbook.loadFromEvents('ob-1.tsv')
orderbook_test = orderbook
orderbook.summary()

# import datetime
# orderbook = Orderbook()
# config = {
#     'startPrice': 10000.0,
#     'endPrice': 9940.0,
#     'levels': 25,
#     'qtyPosition': 0.1,
#     'startTime': datetime.datetime.now(),
#     'duration': datetime.timedelta(minutes=30),
#     'interval': datetime.timedelta(seconds=1)
# }
# orderbook.createArtificial(config)
# orderbook.summary()
#orderbook.plot(show_bidask=True)


import gym_ctc_executioner
env = gym.make("ctc-executioner-v0")
env.configure(orderbook)

#model = loadModel(name='model-sell-artificial-2')
model = loadModel(name='model-sell-imitate-artificial-2')
#model = createModel()
nrTrain = 0
nrTest = 10

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=5000, window_length=1)
# nb_steps_warmup: the default value for that in the DQN OpenAI baselines implementation is 1000
dqn = DQNAgent(model=model, nb_actions=len(env.levels), memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
cbs_train = []
cbs_train = [LivePlotCallback(nb_episodes=50000, avgwindow=20)]
dqn.fit(env, nb_steps=nrTrain, visualize=True, verbose=2, callbacks=cbs_train)
saveModel(model=model, name='model-sell-imitate-artificial-2')

cbs_test = []
cbs_test = [ActionPlotCallback(nb_episodes=nrTest)]
dqn.test(env, nb_episodes=nrTest, visualize=True, verbose=2, callbacks=cbs_test)
