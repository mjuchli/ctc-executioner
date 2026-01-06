# Machine Learning Model & Features Analysis

## Current Feature Structure

### Observation Space
- **Shape**: `(2*lookback+1, bookSize, 2)` for ORDERS features
- **Components**:
  1. **First row**: `[t, i]` - Time remaining and inventory remaining (scalars)
  2. **Next 2*lookback rows**: Historical orderbook states
     - Each state: `(bookSize, 2)` = `(price, size)` for each level
     - Alternating: bids then asks
     - Normalized by best ask price and quantity

### Example Structure
```
Row 0: [t, i, 0, 0, ...]  # Time and inventory
Row 1: [bid_price_1, bid_size_1, ...]  # Bids at t-0
Row 2: [ask_price_1, ask_size_1, ...]  # Asks at t-0
Row 3: [bid_price_1, bid_size_1, ...]  # Bids at t-1
Row 4: [ask_price_1, ask_size_1, ...]  # Asks at t-1
...
```

## Current Model Architectures

### 1. Simple DQN (agent_dqn.py)
```python
Flatten(input) → Dense(bookSize) → Dense(action_size)
```
**Issues:**
- ❌ **Loses temporal structure**: Flattening destroys the time sequence information
- ❌ **Inefficient**: Processes all historical states equally without attention to recency
- ❌ **No feature hierarchy**: Treats time/inventory the same as orderbook features
- ❌ **Too shallow**: Only 2 layers may not capture complex patterns

### 2. LSTM DQN (agent_sb3.py)
```python
LSTM(observations) → Dense(action_size)
```
**Better, but:**
- ✅ Captures temporal dependencies
- ⚠️ **Mixed data types**: LSTM processes `[t, i]` mixed with orderbook features
- ⚠️ **No explicit time/inventory handling**: These critical features are just part of the sequence
- ⚠️ **May not optimize for the specific problem**: Generic LSTM may not be optimal

## Critical Issues

### 1. **Feature Representation Problems**

#### Issue: Time/Inventory Treated as Orderbook Features
- `[t, i]` are scalars with different semantics than price/size
- They're concatenated into the first row, making them hard to distinguish
- The model must learn to separate these conceptually different features

**Better approach:**
- Separate time/inventory as explicit features
- Use them as conditioning inputs or separate branches

#### Issue: Normalization Strategy
- Prices normalized by best ask: `price / bestAsk`
- Sizes normalized by quantity: `size / qty`
- **Problem**: This makes features relative, but loses absolute scale information
- **Problem**: Different normalization for different feature types may confuse the model

#### Issue: Temporal Structure
- Historical states are just stacked vertically
- No explicit temporal encoding (e.g., time deltas, position in sequence)
- Model must infer temporal relationships from position

### 2. **Model Architecture Problems**

#### Issue: No Explicit State Decomposition
The model should understand:
- **Market state**: Orderbook depth, spread, liquidity
- **Execution state**: Time remaining, inventory remaining
- **Action context**: What price level to choose

Current models treat everything as one flat feature vector.

#### Issue: Missing Important Features
- **Spread**: Bid-ask spread (critical for execution)
- **Order book imbalance**: Ratio of bid/ask liquidity
- **Volatility**: Price movement over lookback period
- **Volume profile**: How liquidity is distributed

### 3. **Action Space Issues**

#### Issue: Action is Price Level Offset
- Actions are discrete: `-50` to `+50` levels
- Each level = `0.1 * level` price offset
- **Problem**: Fixed delta doesn't adapt to market conditions
- **Problem**: Same action means different things at different price levels

## Recommendations

### 1. **Feature Engineering Improvements**

#### Separate State Components
```python
observation = {
    'time_inventory': [t, i],  # Explicit scalars
    'orderbook_history': (lookback, bookSize, 2),  # Temporal orderbook
    'market_features': [spread, imbalance, volatility, ...]  # Derived features
}
```

#### Add Derived Features
- **Spread**: `bestAsk - bestBid`
- **Imbalance**: `sum(bid_sizes) / (sum(bid_sizes) + sum(ask_sizes))`
- **Mid-price trend**: Price change over lookback
- **Liquidity concentration**: How much liquidity at best bid/ask vs deeper levels

#### Better Normalization
- Consider z-score normalization for prices
- Use log-scale for sizes (liquidity often log-distributed)
- Keep time/inventory in original scale or normalize separately

### 2. **Model Architecture Improvements**

#### Option A: Multi-Input Network
```python
# Separate branches for different feature types
time_inventory_branch = Dense(32)([t, i])
orderbook_branch = LSTM(128)(orderbook_history)
market_features_branch = Dense(64)(derived_features)

# Concatenate and combine
combined = Concatenate()([time_inventory_branch, orderbook_branch, market_features_branch])
q_values = Dense(action_size)(combined)
```

#### Option B: Attention Mechanism
```python
# Use attention to focus on relevant historical states
attention = Attention()(orderbook_history)
# Combine with time/inventory
combined = Concatenate()([attention, [t, i]])
q_values = Dense(action_size)(combined)
```

#### Option C: Transformer Architecture
- Self-attention over historical orderbook states
- Better at capturing long-range dependencies
- Can learn which historical states matter most

### 3. **Action Space Improvements**

#### Consider Continuous Actions
- Instead of discrete levels, predict price offset directly
- Use actor-critic (e.g., PPO, SAC) instead of DQN
- More flexible, can adapt to market conditions

#### Or: Adaptive Action Space
- Make action space relative to spread
- E.g., actions as multiples of spread: `[-2*spread, -1*spread, 0, +1*spread, +2*spread]`

### 4. **Training Improvements**

#### Curriculum Learning
- Start with simple scenarios (large time window, small inventory)
- Gradually increase difficulty
- Helps model learn basic patterns first

#### Reward Shaping
- Current reward: execution quality
- Consider intermediate rewards for:
  - Partial fills
  - Staying within spread
  - Time management

#### State-Action Value Decomposition
- Instead of just Q(s, a), also predict:
  - Expected fill probability
  - Expected execution time
  - Expected slippage
- Helps model understand why actions are good/bad

## Specific Code Issues

### agent_dqn.py
```python
# Current: Too simple
model.add(Flatten(input_shape=(51, 10, 2)))  # Loses all structure!
model.add(Dense(10))  # Only 10 units for 1020 flattened features
model.add(Dense(101))  # Action size
```

**Problems:**
- 1020 input features → 10 hidden units is a huge bottleneck
- No temporal processing
- Information loss is severe

### agent_sb3.py
```python
# Better, but could improve
LSTMFeatureExtractor:
    - Processes (batch, 51, 10, 2) as temporal sequence
    - But treats [t, i] as just another row in the sequence
```

**Improvements:**
- Extract [t, i] separately
- Use them to condition the LSTM or as additional inputs

## Questions to Consider

1. **Is the lookback window optimal?** (Currently 25 states)
   - Too short: Missing long-term trends
   - Too long: Noise, irrelevant old information

2. **Is bookSize=10 enough?** (Currently 10 levels)
   - May miss deeper liquidity information
   - But more levels = more parameters

3. **Should we use both price AND size?**
   - Current: Both (2 features per level)
   - Could reduce to just price if size not informative
   - Or add more features (e.g., order count)

4. **Is normalization helping or hurting?**
   - Normalization helps training
   - But loses absolute scale information
   - Consider keeping some unnormalized features

## Conclusion

The current model is **functional but suboptimal**:

✅ **What works:**
- LSTM captures some temporal patterns
- Features include relevant information
- Basic structure is sound

❌ **What needs improvement:**
- Feature representation (separate time/inventory)
- Model architecture (multi-input, attention)
- Missing derived features (spread, imbalance)
- Action space (consider continuous or adaptive)

**Priority fixes:**
1. Separate time/inventory from orderbook features
2. Add derived market features (spread, imbalance)
3. Improve model architecture (multi-input or attention)
4. Consider continuous actions or adaptive action space
