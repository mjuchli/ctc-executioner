"""
Enhanced visualization callback for agent training/testing.
Shows order placement, fill status, trades, and reward/loss.
Works with multiple agent frameworks (SB3, DQN, Q-Learning, Keras-RL).
"""

import matplotlib.pyplot as plt
import numpy as np
from ctc_executioner.order_side import OrderSide

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object

try:
    from rl.callbacks import Callback as KerasRLCallback
    KERAS_RL_AVAILABLE = True
except ImportError:
    try:
        from ctc_executioner.agent_utils.callback_base import Callback as KerasRLCallback
        KERAS_RL_AVAILABLE = True
    except ImportError:
        KERAS_RL_AVAILABLE = False
        KerasRLCallback = object


class EnhancedPlotCallback:
    """
    Enhanced visualization callback that shows:
    - Order placement locations
    - Fill status (filled/partial/unfilled)
    - Trade execution details
    - Reward/loss per order
    - Cumulative P&L
    
    Works with:
    - Stable-Baselines3 (inherits from BaseCallback)
    - Keras-RL (inherits from Callback)
    - Custom agents (standalone mode)
    """

    def __init__(self, unwrapped_env, nb_episodes=10, verbose=0):
        # Try to inherit from appropriate base class
        if SB3_AVAILABLE:
            try:
                BaseCallback.__init__(self, verbose)
            except:
                pass
        
        self.unwrapped_env = unwrapped_env
        self.nb_episodes = nb_episodes
        self.episodes = {}
        self.current_episode = {"episode": 0, "steps": {}}
        self.current_step = {}
        self.step_count = 0
        self.episode_count = 0
        self.fig = None
        self.ax = None
        self.verbose = verbose

    def _collect_step_data(self, action=None, reward=None, execution=None):
        """Collect step data from various sources."""
        # Get execution details from environment if available
        if execution is None:
            execution = getattr(self.unwrapped_env, "execution", None)
        
        trades = []
        fill_status = "unfilled"
        qty_executed = 0.0
        qty_remaining = 0.0
        order_price = None
        avg_fill_price = None
        total_profit = 0.0

        if execution:
            trades = execution.getTrades() if hasattr(execution, "getTrades") else []
            qty_executed = execution.getQtyExecuted() if hasattr(execution, "getQtyExecuted") else 0.0
            qty_remaining = execution.getQtyNotExecuted() if hasattr(execution, "getQtyNotExecuted") else 0.0
            
            # Determine fill status
            if qty_executed == 0.0:
                fill_status = "unfilled"
            elif qty_remaining == 0.0:
                fill_status = "filled"
            else:
                fill_status = "partial"
            
            # Get order price
            order = execution.getOrder() if hasattr(execution, "getOrder") else None
            if order:
                order_price = order.getPrice()
            
            # Calculate average fill price and profit
            if trades:
                total_qty = sum(t.getCty() for t in trades)
                if total_qty > 0:
                    avg_fill_price = sum(t.getPrice() * t.getCty() for t in trades) / total_qty
                    # Calculate profit (simplified: difference from order price)
                    if order_price:
                        if self.unwrapped_env.side == OrderSide.BUY:
                            # Profit = (sell_price - buy_price) * qty
                            total_profit = sum((order_price - t.getPrice()) * t.getCty() for t in trades)
                        else:  # SELL
                            # Profit = (sell_price - buy_price) * qty
                            total_profit = sum((t.getPrice() - order_price) * t.getCty() for t in trades)

        return {
            "trades": trades,
            "fill_status": fill_status,
            "qty_executed": qty_executed,
            "qty_remaining": qty_remaining,
            "order_price": order_price,
            "avg_fill_price": avg_fill_price,
            "total_profit": total_profit,
            "reward": reward if reward is not None else 0.0,
        }

    # SB3 compatibility
    def _on_step(self) -> bool:
        """SB3 callback method."""
        actions = self.locals.get("actions", [None])[0] if hasattr(self, 'locals') else None
        rewards = self.locals.get("rewards", [0])[0] if hasattr(self, 'locals') else 0
        
        if isinstance(actions, np.ndarray):
            action = int(actions.item() if actions.size == 1 else actions[0])
        else:
            action = int(actions) if actions is not None else 0

        step_data = self._collect_step_data(action=action, reward=float(rewards))
        
        self.current_step = {
            "action": action,
            "index": getattr(self.unwrapped_env, "orderbookIndex", None),
            "t": getattr(self.unwrapped_env.actionState, "getT", lambda: 0)() if hasattr(self.unwrapped_env, "actionState") else 0,
            "i": getattr(self.unwrapped_env.actionState, "getI", lambda: 0)() if hasattr(self.unwrapped_env, "actionState") else 0,
            **step_data,
        }
        self.current_episode["steps"][self.step_count] = self.current_step
        self.step_count += 1

        dones = self.locals.get("dones", [False]) if hasattr(self, 'locals') else [False]
        if dones[0]:
            self._on_episode_end()
            self.episode_count += 1
            self.step_count = 0
            self.current_episode = {"episode": self.episode_count, "steps": {}}

        return True

    # Keras-RL compatibility
    def on_step_end(self, step, logs):
        """Keras-RL callback method."""
        action = logs.get('action', 0)
        reward = logs.get('reward', 0)
        
        step_data = self._collect_step_data(action=action, reward=reward)
        
        self.current_step = {
            "action": action,
            "index": getattr(self.unwrapped_env, "orderbookIndex", None),
            "t": getattr(self.unwrapped_env.actionState, "getT", lambda: 0)() if hasattr(self.unwrapped_env, "actionState") else 0,
            "i": getattr(self.unwrapped_env.actionState, "getI", lambda: 0)() if hasattr(self.unwrapped_env, "actionState") else 0,
            **step_data,
        }
        self.current_episode["steps"][step] = self.current_step

    def on_episode_begin(self, episode, logs):
        """Keras-RL callback method."""
        self.current_episode = {"episode": episode, "steps": {}}
        self.step_count = 0

    def on_episode_end(self, episode, logs):
        """Keras-RL callback method."""
        if episode == 0:
            self._init_plot()
        self._plot_episode(self.current_episode)
        if episode == (self.nb_episodes - 1):
            if self.fig:
                plt.tight_layout()
                plt.show()
        self.episodes[episode] = self.current_episode

    # Standalone mode (for custom agents)
    def add_step(self, action, reward=None, execution=None, index=None, t=None, i=None):
        """Manually add a step (for custom agents like DQN, Q-Learning)."""
        step_data = self._collect_step_data(action=action, reward=reward, execution=execution)
        
        self.current_step = {
            "action": action,
            "index": index if index is not None else getattr(self.unwrapped_env, "orderbookIndex", None),
            "t": t if t is not None else (getattr(self.unwrapped_env.actionState, "getT", lambda: 0)() if hasattr(self.unwrapped_env, "actionState") else 0),
            "i": i if i is not None else (getattr(self.unwrapped_env.actionState, "getI", lambda: 0)() if hasattr(self.unwrapped_env, "actionState") else 0),
            **step_data,
        }
        self.current_episode["steps"][self.step_count] = self.current_step
        self.step_count += 1

    def end_episode(self):
        """Manually end an episode (for custom agents)."""
        if self.episode_count == 0:
            self._init_plot()
        self._plot_episode(self.current_episode)
        if self.episode_count == (self.nb_episodes - 1):
            if self.fig:
                plt.tight_layout()
                plt.show()
        self.episodes[self.episode_count] = self.current_episode
        self.episode_count += 1
        self.step_count = 0
        self.current_episode = {"episode": self.episode_count, "steps": {}}

    def _on_episode_end(self):
        """Internal method called when episode ends."""
        if self.episode_count == 0:
            self._init_plot()
        self._plot_episode(self.current_episode)
        if self.episode_count == (self.nb_episodes - 1):
            if self.fig:
                plt.tight_layout()
                plt.show()
        self.episodes[self.episode_count] = self.current_episode

    def _init_plot(self):
        """Initialize the plot with orderbook price chart."""
        # Create figure with subplots
        self.fig, self.ax = plt.subplots(figsize=(20, 12))
        self.fig.patch.set_facecolor("#1e1e1e")
        self.ax.set_facecolor("#1e1e1e")
        
        # Get orderbook data
        states = self.unwrapped_env.orderbook.getStates()
        times = [s.getTimestamp() for s in states]
        mid_prices = [s.getBidAskMid() for s in states]
        
        # Plot price line
        self.ax.plot(times, mid_prices, color="#888888", linewidth=1, alpha=0.5, label="Mid Price")
        
        # Plot best bid/ask if available
        try:
            best_bids = [s.getBestBid() for s in states if s.getBuyers()]
            best_asks = [s.getBestAsk() for s in states if s.getSellers()]
            bid_times = [s.getTimestamp() for s in states if s.getBuyers()]
            ask_times = [s.getTimestamp() for s in states if s.getSellers()]
            
            if bid_times:
                self.ax.plot(bid_times, best_bids, color="#00ff88", linewidth=1, alpha=0.7, linestyle="--", label="Best Bid")
            if ask_times:
                self.ax.plot(ask_times, best_asks, color="#ff4444", linewidth=1, alpha=0.7, linestyle="--", label="Best Ask")
        except:
            pass
        
        self.ax.set_xlabel("Time", fontsize=12, color="white")
        self.ax.set_ylabel("Price", fontsize=12, color="white")
        self.ax.set_title("Agent Order Execution Visualization", fontsize=16, fontweight="bold", color="white")
        self.ax.legend(loc="upper left", fontsize=10)
        self.ax.grid(True, alpha=0.3, color="#444444")
        self.ax.tick_params(colors="white")
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')

    def _plot_episode(self, episode):
        """Plot episode with enhanced visualization."""
        if not self.ax:
            return

        steps_data = []
        for key, value in sorted(episode["steps"].items()):
            index = value.get("index")
            if index is None:
                continue
            
            try:
                state = self.unwrapped_env.orderbook.getState(index)
                time = state.getTimestamp()
                mid_price = state.getBidAskMid()
                
                action_val = value.get("action", 0)
                if isinstance(action_val, (np.ndarray, list)):
                    action_val = int(action_val[0] if len(action_val) > 0 else 0)
                else:
                    action_val = int(action_val)
                
                action_delta = 0.1 * self.unwrapped_env.levels[action_val]
                if self.unwrapped_env.side == OrderSide.BUY:
                    order_price = mid_price + action_delta
                else:
                    order_price = mid_price - action_delta
                
                steps_data.append({
                    "time": time,
                    "mid_price": mid_price,
                    "order_price": order_price,
                    "action": action_val,
                    "reward": value.get("reward", 0),
                    "fill_status": value.get("fill_status", "unfilled"),
                    "qty_executed": value.get("qty_executed", 0),
                    "qty_remaining": value.get("qty_remaining", 0),
                    "trades": value.get("trades", []),
                    "avg_fill_price": value.get("avg_fill_price"),
                    "total_profit": value.get("total_profit", 0),
                })
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error plotting step {key}: {e}")
                continue

        if not steps_data:
            return

        # Plot order placements with color coding
        for i, step in enumerate(steps_data):
            # Color based on fill status
            if step["fill_status"] == "filled":
                color = "#00ff88"  # Green
                marker = "o"
                size = 100
            elif step["fill_status"] == "partial":
                color = "#ffaa00"  # Orange/Yellow
                marker = "s"
                size = 80
            else:  # unfilled
                color = "#ff4444"  # Red
                marker = "x"
                size = 60

            # Plot order placement
            self.ax.scatter(
                step["time"],
                step["order_price"],
                c=color,
                marker=marker,
                s=size,
                edgecolors="white",
                linewidths=1,
                zorder=5,
                alpha=0.8,
            )

            # Draw line from order price to mid price (or fill price if available)
            target_price = step["avg_fill_price"] if step["avg_fill_price"] else step["mid_price"]
            line_color = color if step["fill_status"] != "unfilled" else "#666666"
            line_style = "-" if step["fill_status"] == "filled" else "--" if step["fill_status"] == "partial" else ":"
            line_width = 2 if step["fill_status"] == "filled" else 1.5 if step["fill_status"] == "partial" else 1
            
            self.ax.plot(
                [step["time"], step["time"]],
                [step["order_price"], target_price],
                color=line_color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=0.6,
                zorder=3,
            )

            # Annotate with key information
            annotation_parts = []
            if step["fill_status"] == "filled":
                annotation_parts.append("✓ FILLED")
            elif step["fill_status"] == "partial":
                fill_pct = (step["qty_executed"] / (step["qty_executed"] + step["qty_remaining"]) * 100) if (step["qty_executed"] + step["qty_remaining"]) > 0 else 0
                annotation_parts.append(f"⚠ {fill_pct:.0f}% FILLED")
            else:
                annotation_parts.append("✗ UNFILLED")
            
            if step["total_profit"] != 0:
                profit_str = f"+{step['total_profit']:.2f}" if step["total_profit"] > 0 else f"{step['total_profit']:.2f}"
                annotation_parts.append(f"P&L: {profit_str}")
            
            if step["reward"] != 0:
                reward_str = f"+{step['reward']:.2f}" if step["reward"] > 0 else f"{step['reward']:.2f}"
                annotation_parts.append(f"R: {reward_str}")
            
            if annotation_parts:
                annotation = "\n".join(annotation_parts)
                self.ax.annotate(
                    annotation,
                    xy=(step["time"], step["order_price"]),
                    xytext=(10, 20 if i % 2 == 0 else -30),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e1e", edgecolor=color, linewidth=1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1),
                )

        # Update plot
        self.fig.canvas.draw()


# Create SB3-compatible version
if SB3_AVAILABLE:
    class EnhancedPlotCallbackSB3(EnhancedPlotCallback, BaseCallback):
        """SB3-compatible version."""
        pass
else:
    EnhancedPlotCallbackSB3 = EnhancedPlotCallback

# Create Keras-RL-compatible version
if KERAS_RL_AVAILABLE:
    class EnhancedPlotCallbackKerasRL(EnhancedPlotCallback, KerasRLCallback):
        """Keras-RL-compatible version."""
        pass
else:
    EnhancedPlotCallbackKerasRL = EnhancedPlotCallback
