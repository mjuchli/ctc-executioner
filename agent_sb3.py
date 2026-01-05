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
from ctc_executioner.agent_utils.action_plot_callback import ActionPlotCallback
from ctc_executioner.agent_utils.live_plot_callback import LivePlotCallback

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


class ActionPlotCallbackSB3(BaseCallback):
    """
    Stable-Baselines3 compatible version of ActionPlotCallback.
    Adapts the original ActionPlotCallback to work with SB3's callback system.
    """

    def __init__(self, unwrapped_env, nb_episodes=10, verbose=0):
        super(ActionPlotCallbackSB3, self).__init__(verbose)
        self.unwrapped_env = unwrapped_env
        self.nb_episodes = nb_episodes
        self.episodes = {}
        self.current_episode = {"episode": 0, "steps": {}}
        self.current_step = {}
        self.step_count = 0
        self.episode_count = 0
        self.plt = None

    def _on_step(self) -> bool:
        # Get action, observation, reward from locals
        actions = self.locals.get("actions", [None])[0]
        infos = self.locals.get("infos", [{}])

        # Convert action to scalar if it's a numpy array
        if isinstance(actions, np.ndarray):
            action = int(actions.item() if actions.size == 1 else actions[0])
        else:
            action = int(actions) if actions is not None else 0

        # Store step information
        self.current_step = {
            "action": action,
            "index": getattr(self.unwrapped_env, "orderbookIndex", None),
            "t": getattr(self.unwrapped_env.actionState, "getT", lambda: 0)(),
            "i": getattr(self.unwrapped_env.actionState, "getI", lambda: 0)(),
            "reward": float(self.locals.get("rewards", [0])[0]),
        }
        self.current_episode["steps"][self.step_count] = self.current_step
        self.step_count += 1

        # Check if episode ended
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self._on_episode_end()
            self.episode_count += 1
            self.step_count = 0
            self.current_episode = {"episode": self.episode_count, "steps": {}}

        return True

    def _on_episode_end(self):
        """Called when an episode ends."""
        if self.episode_count == 0:
            self.plt = self.unwrapped_env.orderbook.plot(
                show_bidask=True, max_level=0, show=False
            )
        self._plot_episode(self.current_episode)
        if self.episode_count == (self.nb_episodes - 1):
            if self.plt:
                self.plt.show()
        self.episodes[self.episode_count] = self.current_episode

    def _plot_episode(self, episode):
        """Plot episode actions and rewards."""
        from ctc_executioner.order_side import OrderSide

        (
            indices,
            times,
            actions,
            prices,
            order_prices,
            runtimes,
            inventories,
            rewards,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for key, value in episode["steps"].items():
            index = value.get("index")
            if index is None:
                continue
            indices.append(index)
            runtimes.append(value.get("t", 0))
            inventories.append(value.get("i", 0))
            rewards.append(value.get("reward", 0))
            actions.append(value.get("action", 0))
            state = self.unwrapped_env.orderbook.getState(index)
            prices.append(state.getBidAskMid())
            times.append(state.getTimestamp())
            action_val = value.get("action", 0)
            # Ensure action is an integer
            if isinstance(action_val, (np.ndarray, list)):
                action_val = int(action_val[0] if len(action_val) > 0 else 0)
            else:
                action_val = int(action_val)
            action_delta = 0.1 * self.unwrapped_env.levels[action_val]
            if self.unwrapped_env.side == OrderSide.BUY:
                order_prices.append(state.getBidAskMid() + action_delta)
            else:
                order_prices.append(state.getBidAskMid() - action_delta)

        if self.plt and times:
            self.plt.scatter(times, order_prices, s=20)
            for i, time in enumerate(times):
                style = "k-" if (i == 0 or i == len(times) - 1) else "k--"
                self.plt.plot(
                    [time, time],
                    [
                        prices[i] - 0.005 * prices[i],
                        prices[i] + 0.005 * prices[i],
                    ],
                    style,
                    lw=1,
                )
                txt = (
                    "a="
                    + str(self.unwrapped_env.levels[actions[i]])
                    + "\nr="
                    + str(round(rewards[i], 2))
                )
                self.plt.annotate(txt, (times[i], prices[i]))
                txt = "t=" + str(runtimes[i]) + "\ni=" + str(round(inventories[i], 2))
                self.plt.annotate(txt, (times[i], prices[i] - 0.005 * prices[i]))


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

    # Create artificial orderbook for testing
    orderbook = Orderbook()
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

    # Testing with ActionPlotCallback
    print(f"\nTesting for {nrTest} episodes...")
    test_callback = ActionPlotCallbackSB3(unwrapped_env, nb_episodes=nrTest, verbose=1)

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
