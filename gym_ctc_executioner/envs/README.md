# ExecutionEnv Overview

A Gymnasium environment for training RL agents to execute orders in a limit order book.

## Core Concept
The agent learns to place limit orders at different price levels to execute a target quantity over time.

## Key Components

### 1. State Space (Observations)
- **Time remaining (`T`)**: e.g., [0, 10, 20, ..., 100]
- **Inventory remaining (`I`)**: e.g., [0.0, 0.1, 0.2, ..., 1.0]
- **Orderbook features**: bid/ask prices and sizes over a lookback window
- **Shape**: `(2*lookback+1, bookSize, 2)` for ORDERS features

### 2. Action Space
- **Discrete actions**: price level offsets (e.g., -50 to +50)
- Each action maps to a price level relative to the mid-price
- Example: action 0 = -50 levels, action 50 = mid-price, action 100 = +50 levels

### 3. Step Function Flow
```
Agent chooses action (price level)
↓
Create/Update Execution (limit order at that level)
↓
Run execution against orderbook (match orders)
↓
Calculate reward (based on execution quality)
↓
Update state (time, inventory, features)
↓
Return (observation, reward, done, info)
```

### 4. Episode Termination
- Order is fully filled (`execution.isFilled()`)
- Inventory reaches 0 (`i == 0`)

### 5. Reward
- **End of episode**: total reward from execution
- **During episode**: weighted reward based on partial fills

## Configuration Parameters

- `T`: Time steps (e.g., `(0, 100, 10)` = [0, 10, 20, ..., 100])
- `I`: Inventory levels (e.g., `(0, 1, 0.1)` = [0.0, 0.1, ..., 1.0])
- `levels`: Price level offsets (e.g., `(-50, 50, 1)` = 101 actions)
- `lookback`: Historical orderbook window (default: 25)
- `bookSize`: Number of price levels to observe (default: 10)
- `side`: BUY or SELL order

## Example Flow

1. **Reset**: Start with max time (100) and max inventory (1.0) at a random orderbook state
2. **Agent action**: Chooses price level (e.g., action 50 = mid-price)
3. **Execution**: Places limit order, matches against orderbook over runtime
4. **Update**: Reduces time and inventory based on what was executed
5. **Repeat**: Until order is filled or inventory is 0

The agent learns which price levels to use at different times and inventory levels to maximize execution quality (minimize slippage, maximize fill rate).
