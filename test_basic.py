#!/usr/bin/env python3
"""
Basic test script to verify the codebase works with updated dependencies.
"""

import numpy as np
import gymnasium as gym
import gym_ctc_executioner
from ctc_executioner.orderbook import Orderbook
from ctc_executioner.order_side import OrderSide
from ctc_executioner.feature_type import FeatureType
import datetime


def test_environment_creation():
    """Test if we can create and configure the environment."""
    print("Testing environment creation...")

    # Create artificial orderbook for testing
    orderbook = Orderbook()
    config = {
        "startPrice": 10000.0,
        "priceFunction": lambda p0, s, samples: p0
        + 10 * np.sin(2 * np.pi * 10 * (s / samples)),
        "levels": 50,
        "qtyPosition": 0.1,
        "startTime": datetime.datetime.now(),
        "duration": datetime.timedelta(seconds=1000),  # More states for lookback
        "interval": datetime.timedelta(seconds=1),
    }
    orderbook.createArtificial(config)
    print(f"✓ Orderbook created with {len(orderbook.states)} states")

    # Create environment
    env = gym.make("ctc-executioner-v0")
    # Unwrap if needed (Gymnasium wraps environments)
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env

    unwrapped_env._configure(
        orderbook=orderbook,
        side=OrderSide.SELL,
        featureType=FeatureType.ORDERS,
        lookback=25,  # Explicit lookback
        bookSize=10,
    )
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")

    return env, unwrapped_env


def test_environment_reset():
    """Test environment reset."""
    print("\nTesting environment reset...")
    env, unwrapped_env = test_environment_creation()

    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Info: {info}")

    return env, unwrapped_env, obs


def test_environment_step():
    """Test environment step."""
    print("\nTesting environment step...")
    env, unwrapped_env, obs = test_environment_reset()

    # Take a random action
    action = env.action_space.sample()
    obs_next, reward, terminated, truncated, info = env.step(action)

    print(f"✓ Step successful")
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}, Truncated: {truncated}")
    print(f"  Next observation shape: {obs_next.shape}")

    return env, unwrapped_env


def test_keras_model():
    """Test if we can create a simple Keras model."""
    print("\nTesting Keras model creation...")
    from keras.models import Sequential
    from keras.layers import Dense, Flatten
    from keras import optimizers

    # Create a simple model
    model = Sequential()
    model.add(Flatten(input_shape=(10, 10, 2)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="linear"))
    model.compile(optimizers.Adam(learning_rate=0.001), loss="mse")

    print(f"✓ Keras model created")
    print(f"  Model summary:")
    model.summary()

    return model


def test_dqn_agent_creation():
    """Test if we can create a DQN agent."""
    print("\nTesting DQN agent creation...")
    # Import only the class, not the module-level code
    import sys
    import importlib.util

    # Load the module without executing the main code
    spec = importlib.util.spec_from_file_location("agent_dqn", "agent_dqn.py")
    agent_dqn_module = importlib.util.module_from_spec(spec)
    # Don't execute the module to avoid file loading
    # Instead, we'll create the agent class directly
    from agent_dqn import AgentDQN

    env, unwrapped_env = test_environment_creation()
    agent = AgentDQN(env)

    print(f"✓ DQN agent created")
    print(f"  Action size: {agent.action_size}")
    print(f"  Model built successfully")

    return agent, env, unwrapped_env


def test_dqn_agent_prediction():
    """Test if DQN agent can make predictions."""
    print("\nTesting DQN agent prediction...")
    agent, env, unwrapped_env = test_dqn_agent_creation()

    obs, _ = env.reset()
    # Add batch dimension
    obs_batch = np.expand_dims(obs, axis=0)

    action = agent.act(obs_batch)
    print(f"✓ Agent prediction successful")
    print(f"  Selected action: {action}")

    return agent, env, unwrapped_env


if __name__ == "__main__":
    print("=" * 60)
    print("Basic Functionality Test")
    print("=" * 60)

    try:
        # Run tests (skip DQN tests that require file loading)
        test_environment_creation()
        test_environment_reset()
        test_environment_step()
        test_keras_model()
        # Skip DQN tests for now as they require data files
        # test_dqn_agent_creation()
        # test_dqn_agent_prediction()

        print("\n" + "=" * 60)
        print("✓ Core functionality tests passed!")
        print("  (DQN agent tests skipped - require data files)")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
