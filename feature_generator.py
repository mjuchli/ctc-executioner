from orderbook import Orderbook
import numpy as np
import pandas as pd
import os

book = 'query_result_test.tsv'
tmp='feature.tsv'
orderbook = Orderbook()
orderbook.loadFromFile(book)
states = orderbook.getStates()


def stateDiff(start, end):
    """Calculate time difference between two states."""
    consumed = (end.getTimestamp() - start.getTimestamp()).total_seconds()
    return consumed


def getPastState(i, t):
    """Find state at index i - time t."""
    endState = states[i]
    state = endState
    while(stateDiff(state, endState) < t):
        i = i - 1
        if i < 0:
            raise Exception("Not enough states available for diff.")
        state = states[i]
    return i


def traverse(f, g, default=0.0, t=60):
    """Traverse states and apply g(i, f(i-t, i)) for states at time e.g. i."""
    startState = states[0]
    consumed = 0.0
    for i in range(len(states)):
        state = states[i]
        consumed = stateDiff(startState, state)
        # print("consumed: " + str(consumed) + " at i=" + str(i))
        if consumed < t:
            g(i, default)
        else:
            pastState = getPastState(i, t)
            g(i, f(pastState, i+1))


def calcVolume(start, end):
    """Calculate volume for range of states."""
    vol = 0.0
    for j in range(start, end):
        tempState = states[j]
        vol = vol + tempState.getVolume()
    return vol


def calcStdPrice(start, end):
    """Calculate standard deviation for prices in range of states."""
    prices = map(lambda x: states[x].getTradePrice(), range(start, end))
    return np.std(list(prices))


def calcMeanPrice(start, end):
    """Calculate mean for prices in range of states."""
    prices = map(lambda x: states[x].getTradePrice(), range(start, end))
    return np.mean(list(prices))


def toFile(i, x):
    """Write feature to temp file."""
    output = str(x) + '\n'
    with open(tmp, 'a') as fa:
        fa.write(output)


def concatFeature(f, default=0.0, t=60):
    """Appends feature f as a new column to orderbook."""
    traverse(f, toFile, default, t)
    # Append feature column
    df1 = pd.read_csv(book, sep='\t')
    df2 = pd.read_csv(tmp, sep='\t')
    concat_df = pd.concat([df1, df2], axis=1)
    # Overwrite book
    concat_df.to_csv(book, sep='\t', header=False, index=False, float_format='%.8f')
    # Cleanup tmp
    os.remove(tmp)


def printFeature(f, default=0.0, t=60):
    """Prints feature f."""
    traverse(f, lambda i, x: print(str((i, x))), default, t)


# printFeature(calcMeanPrice)
# concatFeature(calcMeanPrice)
# concatFeature(calcVolume)
concatFeature(calcStdPrice)
