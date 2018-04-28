import unittest
from qlearn import QLearn
from action_state import ActionState
import numpy as np


class QlearnTest(unittest.TestCase):
    def testStateEquality(self):
        ai = QLearn([-1, 0, 1])
        a1 = ActionState(1.0, 1.0, {'vol60': 1})
        a2 = ActionState(1.0, 1.0, {'vol60': 1})
        ai.learn(a1, 1, 1.0, a2)
        self.assertEqual(ai.getQAction(a2), 1)

    #def testQTableLookup(self):
actions = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -7, -10, -15, -20]
ai = QLearn(actions)
ai.q = np.load('test_q.npy').item()
ai.q
state = ActionState(30, 0.9, {})
ai.q.get((state, -10))
print(ai.getQAction(state))
