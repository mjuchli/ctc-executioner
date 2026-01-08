# Order placement with Reinforcement Learning 

CTC-Executioner is a tool that provides an on-demand execution/placement strategy for limit orders on crypto currency markets using Reinforcement Learning techniques. The underlying framework provides functionalities which allow to analyse order book data and derive features thereof. Those findings can then be used in order to dynamically update the decision making process of the execution strategy.

The methods being used are based on a research project (master thesis) currently proceeding at TU Delft.

## Requirements

- **Python**: 3.10 or higher (tested with Python 3.13.7)
- **Gymnasium**: 1.0.0 or higher (migrated from OpenAI Gym)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Agent

### 1. Activate the Virtual Environment

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 2. Verify Dependencies

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `stable-baselines3>=2.0.0`
- `gymnasium>=1.0.0`
- `tensorflow>=2.15.0`
- `keras>=3.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`

### 3. Run the Agent

#### Option A: Run the Stable-Baselines3 Agent (Recommended)

```bash
python agent_sb3.py
```

This will:
- Create an artificial orderbook for testing
- Train a DQN model for 100,000 steps
- Save the model to `models/dqn_ctc_executioner_sb3.zip`
- Test the model for 10 episodes
- Display action plots if configured

#### Option B: Run the Custom DQN Agent

```bash
python agent_dqn.py
```

**Note:** This requires data files. Make sure `data/events/ob-1.tsv` exists or modify the file path in the script.

#### Option C: Run Tests

```bash
python test_basic.py
```

This runs basic functionality tests to verify the environment works.

## Configuration

### Modify Training Parameters

Edit `agent_sb3.py` to change:

```python
nrTrain = 100000    # Number of training steps
nrTest = 10         # Number of test episodes
model_name = "dqn_ctc_executioner_sb3"  # Model save name
```

### Use Real Data

Replace the artificial orderbook with real data:

```python
# Instead of:
orderbook.createArtificial(config)

# Use:
orderbook.loadFromEvents('data/events/ob-train.tsv')
```

### Adjust Model Architecture

The LSTM feature extractor can be modified in the `LSTMFeatureExtractor` class:

```python
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=512):
        # Change features_dim to adjust LSTM hidden size
        ...
```

## Model Management

### Load a Saved Model

Models are automatically saved after training. To load an existing model:

```python
model = load_sb3_model("dqn_ctc_executioner_sb3", env)
```

### Model Location

Models are saved in the `models/` directory:
- `models/dqn_ctc_executioner_sb3.zip` - Full model (policy + replay buffer)

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'stable_baselines3'"

**Solution:**
```bash
pip install stable-baselines3
```

### Issue: "AttributeError: 'OrderEnforcing' object has no attribute 'setOrderbook'"

**Solution:** The environment is wrapped. Use `env.unwrapped`:
```python
unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
unwrapped_env.setOrderbook(orderbook)
```

### Issue: "FileNotFoundError: data/events/ob-1.tsv"

**Solution:** Either:
1. Use artificial data (already configured in `agent_sb3.py`)
2. Download/place data files in `data/events/`
3. Modify the file path in the script

### Issue: Training is slow

**Solution:** 
- Reduce `nrTrain` for testing
- Use GPU if available (Stable-Baselines3 will automatically use it if PyTorch CUDA is installed)
- Reduce `buffer_size` in model creation

## Deactivate Virtual Environment

When done:

```bash
deactivate
```

## Documentation

Comprehensive documentation and concepts explained in the [academic report](https://github.com/backender/ctc-executioner/blob/master/docs/report.pdf)

For hands-on documentation and examples see [Wiki](https://github.com/backender/ctc-executioner/wiki)

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
import gymnasium as gym
import gym_ctc_executioner
env = gym.make("ctc-executioner-v0")
env.setOrderbook(orderbook)

env_test = gym.make("ctc-executioner-v0")
env_test.setOrderbook(orderbook_test)
```
