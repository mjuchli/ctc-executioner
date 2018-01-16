from orderbook import Orderbook

orderbook = Orderbook()
orderbook.loadFromFile('query_result_small.tsv')
states = orderbook.getStates()


def stateDiff(start, end):
    consumed = (end.getTimestamp() - start.getTimestamp()).total_seconds()
    return consumed


def getPastState(i, t):
    endState = states[i]
    state = endState
    while(stateDiff(state, endState) < t):
        i = i - 1
        if i < 0:
            raise Exception("Not enough states available for diff.")
        state = states[i]
    return i

def getVolume(t=60):
    """Calculate volume for the last t seconds."""
    volumes = []
    startState = states[0]
    consumed = 0.0
    for i in range(len(states)):
        state = states[i]
        consumed = stateDiff(startState, state)
        # print("consumed: " + str(consumed) + " at i=" + str(i))
        if consumed < t:
            volumes.append(0.0)
            # print(0.0)
        else:
            pastState = getPastState(i, t)
            vol = 0.0
            for j in range(pastState, i+1):
                tempState = states[j]
                vol = vol + tempState.getVolume()
            volumes.append(vol)
            # print(vol)
    return volumes

print(getVolume())
