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
        
        # Track cumulative P/L across all episodes
        self.cumulative_profit = 0.0
        self.cumulative_reward = 0.0
        self.total_trades = 0
        self.total_filled = 0
        self.total_partial = 0
        self.total_unfilled = 0
        self.pl_text = None
        
        # Initialize plot immediately for real-time streaming (if orderbook is available)
        try:
            self._init_plot()
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)  # Show plot without blocking
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not initialize plot immediately: {e}")
            # Will try again on first step

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
        
        # Update cumulative stats
        self._update_cumulative_stats(step_data)
        
        # Plot this step immediately for real-time streaming
        self._plot_step(self.current_step)
        
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
        
        # Update cumulative stats
        self._update_cumulative_stats(step_data)
        
        # Plot this step immediately for real-time streaming
        self._plot_step(self.current_step)
        
        self.step_count += 1

    def end_episode(self):
        """Manually end an episode (for custom agents)."""
        # Update P/L summary display
        self._update_pl_summary()
        # Force redraw
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        self.episodes[self.episode_count] = self.current_episode
        self.episode_count += 1
        self.step_count = 0
        self.current_episode = {"episode": self.episode_count, "steps": {}}
        
        # Show plot at the end of all episodes
        if self.episode_count == (self.nb_episodes - 1):
            if self.fig:
                plt.ioff()  # Turn off interactive mode
                plt.tight_layout()
                plt.show(block=True)  # Block until closed

    def _on_episode_end(self):
        """Internal method called when episode ends."""
        # Update P/L summary display
        self._update_pl_summary()
        # Force redraw
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        self.episodes[self.episode_count] = self.current_episode
        
        # Show plot at the end of all episodes
        if self.episode_count == (self.nb_episodes - 1):
            if self.fig:
                plt.ioff()  # Turn off interactive mode
                plt.tight_layout()
                plt.show(block=True)  # Block until closed

    def _init_plot(self):
        """Initialize the plot with orderbook price chart."""
        if self.fig is not None:
            return  # Already initialized
        
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
        self.ax.set_title("Agent Order Execution Visualization (Real-time)", fontsize=16, fontweight="bold", color="white")
        self.ax.legend(loc="upper left", fontsize=10)
        self.ax.grid(True, alpha=0.3, color="#444444")
        self.ax.tick_params(colors="white")
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        # Initialize P/L summary text (will be updated)
        self.pl_text = None
        self._update_pl_summary()
        
        # Draw initial plot
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_cumulative_stats(self, step_data):
        """Update cumulative statistics across all episodes."""
        self.cumulative_profit += step_data.get("total_profit", 0)
        self.cumulative_reward += step_data.get("reward", 0)
        
        fill_status = step_data.get("fill_status", "unfilled")
        if fill_status == "filled":
            self.total_filled += 1
        elif fill_status == "partial":
            self.total_partial += 1
        else:
            self.total_unfilled += 1
        
        if step_data.get("trades"):
            self.total_trades += len(step_data.get("trades", []))

    def _update_pl_summary(self):
        """Update and display P/L summary text box."""
        if not self.ax:
            return
        
        # Remove old text if exists
        if self.pl_text:
            self.pl_text.remove()
        
        # Create summary text
        summary_lines = [
            "P/L Summary (All Episodes)",
            f"Total P&L: {self.cumulative_profit:+.2f}",
            f"Total Reward: {self.cumulative_reward:+.2f}",
            f"Episodes: {self.episode_count}/{self.nb_episodes}",
            "",
            "Order Status:",
            f"  Filled: {self.total_filled}",
            f"  Partial: {self.total_partial}",
            f"  Unfilled: {self.total_unfilled}",
            f"  Total Trades: {self.total_trades}",
        ]
        
        summary_text = "\n".join(summary_lines)
        
        # Determine text color based on P/L
        text_color = "#00ff88" if self.cumulative_profit >= 0 else "#ff4444"
        
        # Place summary in upper right corner
        self.pl_text = self.ax.text(
            0.98, 0.98,
            summary_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            color=text_color,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#1e1e1e",
                edgecolor=text_color,
                linewidth=2,
                alpha=0.9
            ),
            family='monospace',
        )

    def _plot_step(self, step_data):
        """Plot a single step immediately for real-time streaming."""
        if not self.ax:
            return
        
        index = step_data.get("index")
        if index is None:
            return
        
        try:
            state = self.unwrapped_env.orderbook.getState(index)
            time = state.getTimestamp()
            mid_price = state.getBidAskMid()
            
            action_val = step_data.get("action", 0)
            if isinstance(action_val, (np.ndarray, list)):
                action_val = int(action_val[0] if len(action_val) > 0 else 0)
            else:
                action_val = int(action_val)
            
            action_delta = 0.1 * self.unwrapped_env.levels[action_val]
            if self.unwrapped_env.side == OrderSide.BUY:
                order_price = mid_price + action_delta
            else:
                order_price = mid_price - action_delta
            
            step = {
                "time": time,
                "mid_price": mid_price,
                "order_price": order_price,
                "action": action_val,
                "reward": step_data.get("reward", 0),
                "fill_status": step_data.get("fill_status", "unfilled"),
                "qty_executed": step_data.get("qty_executed", 0),
                "qty_remaining": step_data.get("qty_remaining", 0),
                "trades": step_data.get("trades", []),
                "avg_fill_price": step_data.get("avg_fill_price"),
                "total_profit": step_data.get("total_profit", 0),
            }
        except Exception as e:
            if self.verbose > 0:
                print(f"Error plotting step: {e}")
            return

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

        # Annotate with key information (same detailed format as before)
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
            # Use alternating offset to avoid overlap
            step_idx = len([s for s in self.current_episode["steps"].values() if s.get("index") is not None])
            self.ax.annotate(
                annotation,
                xy=(step["time"], step["order_price"]),
                xytext=(10, 20 if step_idx % 2 == 0 else -30),
                textcoords="offset points",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e1e", edgecolor=color, linewidth=1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

        # Update P/L summary
        self._update_pl_summary()
        
        # Update plot immediately
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
