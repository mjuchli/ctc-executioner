"""
Stable-Baselines3 DQN Agent for CTC Executioner
Migrated from keras-rl to Stable-Baselines3
"""

import logging
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy
import torch
import torch.nn as nn

from ctc_executioner.order_side import OrderSide
from ctc_executioner.orderbook import Orderbook
from ctc_executioner.agent_utils.enhanced_plot_callback import EnhancedPlotCallback

import gymnasium as gym
import gym_ctc_executioner
import datetime

# logging.basicConfig(level=logging.INFO)


class LSTMFeatureExtractor(nn.Module):
    """
    Custom LSTM feature extractor matching the original keras-rl model architecture.
    """

    def __init__(self, observation_space, features_dim=512):
        super(LSTMFeatureExtractor, self).__init__()
        # observation_space.shape is (51, 10, 2)
        # Reshape to (51, 20) for LSTM input
        if len(observation_space.shape) == 3:
            input_size = observation_space.shape[1] * observation_space.shape[2]
        else:
            # Fallback if shape is different
            input_size = (
                observation_space.shape[-1]
                if len(observation_space.shape) > 1
                else observation_space.shape[0]
            )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=features_dim,
            batch_first=True,
        )
        self.features_dim = features_dim

    def forward(self, observations):
        # observations shape: (batch, 51, 10, 2) or (batch, time_steps, features)
        batch_size = observations.shape[0]
        # Handle different input shapes
        if len(observations.shape) == 4:
            # (batch, time, height, width) -> (batch, time, height*width)
            obs_reshaped = observations.view(
                batch_size,
                observations.shape[1],
                observations.shape[2] * observations.shape[3],
            )
        elif len(observations.shape) == 3:
            # Already in (batch, time, features) format
            obs_reshaped = observations
        else:
            # Flatten if needed
            obs_reshaped = observations.view(batch_size, 1, -1)

        # LSTM forward
        lstm_out, _ = self.lstm(obs_reshaped)
        # Take the last output
        return lstm_out[:, -1, :]  # (batch, features_dim)


class CustomDQNPolicy(DQNPolicy):
    """
    Custom DQN Policy with LSTM feature extractor.
    """

    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=LSTMFeatureExtractor,
            features_extractor_kwargs={"features_dim": 512},
        )


class EpisodeRewardCallback(BaseCallback):
    """Callback to track episode rewards."""

    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # Get reward from info dict if available
        infos = self.locals.get("infos", [{}])
        if infos and "episode" in infos[0]:
            episode_info = infos[0]["episode"]
            self.episode_reward = episode_info["r"]
            self.episode_rewards.append(self.episode_reward)
            if self.verbose > 0:
                print(f"Episode reward: {self.episode_reward}")
        return True


def create_sb3_model(env, model_path=None):
    """
    Create a Stable-Baselines3 DQN model.

    Args:
        env: Gymnasium environment
        model_path: Optional path to load existing model

    Returns:
        DQN model
    """
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}")
        model = DQN.load(model_path, env=env)
    else:
        print("Creating new DQN model...")
        # Create model with custom policy
        model = DQN(
            CustomDQNPolicy,
            env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=100,
            batch_size=32,
            tau=0.01,  # target network update rate (equivalent to target_model_update)
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            target_update_interval=100,  # Update target network every 100 steps
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
        )
    return model


def save_sb3_model(model, name):
    """Save Stable-Baselines3 model."""
    model_path = f"models/{name}"
    model.save(model_path)
    print(f'Saved model to "{model_path}.zip"')


def load_sb3_model(name, env):
    """Load Stable-Baselines3 model."""
    model_path = f"models/{name}"
    model = DQN.load(model_path, env=env)
    print(f'Loaded model from "{model_path}.zip"')
    return model


# Main execution
if __name__ == "__main__":
    import os

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load orderbook - try real data first, fallback to artificial
    orderbook = Orderbook()
    try:
        orderbook.loadFromEvents("data/events/ob-train.tsv")
        print("Loaded orderbook from data/events/ob-train.tsv")
    except (FileNotFoundError, IOError):
        print("Data file not found, creating artificial orderbook...")
        config = {
            "startPrice": 10000.0,
            "priceFunction": lambda p0, s, samples: p0
            + 10 * np.sin(2 * np.pi * 10 * (s / samples)),
            "levels": 50,
            "qtyPosition": 0.1,
            "startTime": datetime.datetime.now(),
            "duration": datetime.timedelta(seconds=1000),
            "interval": datetime.timedelta(seconds=1),
        }
        orderbook.createArtificial(config)
    orderbook.summary()

    # Create environment
    env = gym.make("ctc-executioner-v0")
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    unwrapped_env.setOrderbook(orderbook)

    # Wrap environment for vectorization (required by SB3)
    env = DummyVecEnv([lambda: env])

    # Training parameters
    nrTrain = 100
    nrTest = 10
    model_name = "dqn_ctc_executioner_sb3"

    # Create or load model
    try:
        model = load_sb3_model(model_name, env)
        print("Using loaded model")
    except (FileNotFoundError, ValueError):
        print("Model file not found, creating new model...")
        model = create_sb3_model(env)

    # Training
    print(f"\nTraining for {nrTrain} steps...")
    callback = EpisodeRewardCallback(verbose=1)
    model.learn(
        total_timesteps=nrTrain,
        callback=callback,
        log_interval=10,
    )

    # Save model after training
    save_sb3_model(model, model_name)

    # Testing with Enhanced Plot Callback (shows order placement, fills, and rewards)
    print(f"\nTesting for {nrTest} episodes...")
    print(
        "Using Enhanced Plot Callback - shows order placement, fill status, and reward/loss"
    )
    test_callback = EnhancedPlotCallback(unwrapped_env, nb_episodes=nrTest, verbose=1)

    obs = env.reset()
    episode_rewards = []
    episode_count = 0

    while episode_count < nrTest:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Update callback manually for testing
        test_callback.locals = {
            "actions": [action],
            "rewards": [reward],
            "dones": [done],
            "infos": info,
        }
        test_callback._on_step()

        if done[0]:
            episode_count += 1
            if info and len(info) > 0 and "episode" in info[0]:
                episode_reward = info[0]["episode"]["r"]
                episode_rewards.append(episode_reward)
                print(f"Episode {episode_count} reward: {episode_reward}")
            obs = env.reset()

    if episode_rewards:
        print(f"\nAverage test reward: {np.mean(episode_rewards):.2f}")
        print(f"Std test reward: {np.std(episode_rewards):.2f}")
