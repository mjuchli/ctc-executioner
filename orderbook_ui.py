"""Interactive Orderbook UI

A visualization tool for viewing orderbook states that looks like a real exchange orderbook.
Allows traversing through timestamps to see how the orderbook changes over time.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from ctc_executioner.orderbook import Orderbook
import numpy as np
from datetime import datetime


class OrderbookUI:
    """Interactive orderbook visualization UI."""

    def __init__(self, orderbook):
        self.orderbook = orderbook
        self.current_index = 0
        self.max_levels = 20  # Number of levels to display on each side
        self.fig = None
        self.ax = None
        self.bid_table = None
        self.ask_table = None
        self.info_text = None

    def _format_price(self, price):
        """Format price for display."""
        return f"{price:,.2f}"

    def _format_size(self, size):
        """Format size/quantity for display."""
        if size >= 1000:
            return f"{size/1000:.2f}K"
        return f"{size:.4f}"

    def _get_orderbook_data(self, state_index):
        """Extract orderbook data for display."""
        state = self.orderbook.getState(state_index)
        buyers = state.getBuyers()
        sellers = state.getSellers()

        # Get best bid and ask
        best_bid = state.getBestBid() if buyers else 0
        best_ask = state.getBestAsk() if sellers else 0
        spread = best_ask - best_bid if (best_ask and best_bid) else 0
        mid_price = state.getBidAskMid() if (buyers and sellers) else 0

        # Prepare bid data (sorted descending, limit to max_levels)
        bid_data = []
        cumulative_bid = 0
        for i, entry in enumerate(buyers[: self.max_levels]):
            cumulative_bid += entry.getQty()
            bid_data.append(
                {
                    "price": entry.getPrice(),
                    "size": entry.getQty(),
                    "cumulative": cumulative_bid,
                }
            )

        # Prepare ask data (sorted ascending, limit to max_levels)
        ask_data = []
        cumulative_ask = 0
        for i, entry in enumerate(sellers[: self.max_levels]):
            cumulative_ask += entry.getQty()
            ask_data.append(
                {
                    "price": entry.getPrice(),
                    "size": entry.getQty(),
                    "cumulative": cumulative_ask,
                }
            )

        return (
            bid_data,
            ask_data,
            best_bid,
            best_ask,
            spread,
            mid_price,
            state.getTimestamp(),
        )

    def _create_orderbook_display(self):
        """Create the orderbook visualization."""
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.patch.set_facecolor("#1e1e1e")  # Dark background like exchanges
        self.ax.set_facecolor("#1e1e1e")
        self.fig.suptitle(
            "Orderbook Viewer", fontsize=16, fontweight="bold", color="white"
        )
        self.ax.axis("off")

        # Create main container with dark theme
        main_rect = mpatches.FancyBboxPatch(
            (0.05, 0.1),
            0.9,
            0.85,
            boxstyle="round,pad=0.01",
            edgecolor="#333333",
            facecolor="#2d2d2d",
            linewidth=1,
            transform=self.fig.transFigure,
        )
        self.ax.add_patch(main_rect)

        # Update display
        self._update_display()

        # Add slider for timestamp navigation
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        max_index = len(self.orderbook.getStates()) - 1
        self.slider = Slider(
            ax_slider,
            "Timestamp",
            0,
            max_index,
            valinit=self.current_index,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self._on_slider_change)

        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.05, 0.03])
        ax_next = plt.axes([0.82, 0.02, 0.05, 0.03])
        self.btn_prev = Button(ax_prev, "◀")
        self.btn_next = Button(ax_next, "▶")
        self.btn_prev.on_clicked(self._prev_state)
        self.btn_next.on_clicked(self._next_state)

        # Add chart plot button
        ax_chart = plt.axes([0.88, 0.02, 0.08, 0.03])
        self.btn_chart = Button(ax_chart, "Chart")
        self.btn_chart.on_clicked(self._show_chart)

        # Add keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _update_display(self):
        """Update the orderbook display with current state."""
        if self.ax is None:
            return

        self.ax.clear()
        self.ax.axis("off")

        bid_data, ask_data, best_bid, best_ask, spread, mid_price, timestamp = (
            self._get_orderbook_data(self.current_index)
        )

        # Header
        header_y = 0.92
        self.ax.text(
            0.5,
            header_y,
            f"Orderbook - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="white",
            transform=self.fig.transFigure,
        )

        # Market info with color coding
        info_y = 0.88
        self.ax.text(
            0.2,
            info_y,
            f"Best Bid: {self._format_price(best_bid)}",
            ha="left",
            va="top",
            fontsize=11,
            color="#00ff88",
            fontweight="bold",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.5,
            info_y,
            f"Mid: {self._format_price(mid_price)}",
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.8,
            info_y,
            f"Best Ask: {self._format_price(best_ask)}",
            ha="right",
            va="top",
            fontsize=11,
            color="#ff4444",
            fontweight="bold",
            transform=self.fig.transFigure,
        )

        # Spread info
        spread_bps = (spread / best_ask * 10000) if best_ask > 0 else 0
        self.ax.text(
            0.5,
            info_y - 0.02,
            f"Spread: {self._format_price(spread)} ({spread_bps:.2f} bps)",
            ha="center",
            va="top",
            fontsize=10,
            color="#ffaa00",
            transform=self.fig.transFigure,
        )

        # Column headers with background
        header_y = 0.82
        header_bg = mpatches.Rectangle(
            (0.06, header_y - 0.015),
            0.88,
            0.03,
            transform=self.fig.transFigure,
            facecolor="#333333",
            edgecolor="#555555",
            linewidth=1,
        )
        self.ax.add_patch(header_bg)

        self.ax.text(
            0.15,
            header_y,
            "BIDS (BUY)",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#00ff88",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.5,
            header_y,
            "PRICE",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.85,
            header_y,
            "ASKS (SELL)",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#ff4444",
            transform=self.fig.transFigure,
        )

        # Sub-headers
        sub_header_y = 0.79
        self.ax.text(
            0.08,
            sub_header_y,
            "Size",
            ha="left",
            va="top",
            fontsize=10,
            color="#aaaaaa",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.22,
            sub_header_y,
            "Total",
            ha="left",
            va="top",
            fontsize=10,
            color="#aaaaaa",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.78,
            sub_header_y,
            "Total",
            ha="left",
            va="top",
            fontsize=10,
            color="#aaaaaa",
            transform=self.fig.transFigure,
        )
        self.ax.text(
            0.92,
            sub_header_y,
            "Size",
            ha="left",
            va="top",
            fontsize=10,
            color="#aaaaaa",
            transform=self.fig.transFigure,
        )

        # Draw orderbook levels
        start_y = 0.75
        row_height = 0.025
        max_rows = min(len(bid_data), len(ask_data), self.max_levels)

        for i in range(max_rows):
            y_pos = start_y - i * row_height

            # Row background (alternating for readability)
            row_bg_color = "#252525" if i % 2 == 0 else "#2d2d2d"
            row_rect = mpatches.Rectangle(
                (0.06, y_pos - 0.01),
                0.88,
                row_height,
                transform=self.fig.transFigure,
                facecolor=row_bg_color,
                edgecolor="#333333",
                linewidth=0.3,
            )
            self.ax.add_patch(row_rect)

            # Bid side (left)
            if i < len(bid_data):
                bid = bid_data[i]
                # Highlight best bid with green background
                is_best_bid = i == 0 and bid["price"] == best_bid
                if is_best_bid:
                    highlight_rect = mpatches.Rectangle(
                        (0.06, y_pos - 0.01),
                        0.18,
                        row_height,
                        transform=self.fig.transFigure,
                        facecolor="#004422",
                        edgecolor="#00ff88",
                        linewidth=1,
                    )
                    self.ax.add_patch(highlight_rect)

                self.ax.text(
                    0.08,
                    y_pos,
                    self._format_size(bid["size"]),
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="#00ff88" if is_best_bid else "#aaaaaa",
                    fontweight="bold" if is_best_bid else "normal",
                    transform=self.fig.transFigure,
                )
                self.ax.text(
                    0.22,
                    y_pos,
                    self._format_size(bid["cumulative"]),
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="#888888",
                    transform=self.fig.transFigure,
                )

            # Price (center) - show bid price on left side, ask price on right side
            # This creates a typical exchange orderbook view where prices are shown twice
            if i < len(bid_data):
                bid_price = bid_data[i]["price"]
                is_best_bid_price = i == 0 and bid_price == best_bid
                self.ax.text(
                    0.42,
                    y_pos,
                    self._format_price(bid_price),
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if is_best_bid_price else "normal",
                    color="#00ff88" if is_best_bid_price else "#aaaaaa",
                    transform=self.fig.transFigure,
                )

            if i < len(ask_data):
                ask_price = ask_data[i]["price"]
                is_best_ask_price = i == 0 and ask_price == best_ask
                self.ax.text(
                    0.58,
                    y_pos,
                    self._format_price(ask_price),
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if is_best_ask_price else "normal",
                    color="#ff4444" if is_best_ask_price else "#aaaaaa",
                    transform=self.fig.transFigure,
                )

            # Ask side (right)
            if i < len(ask_data):
                ask = ask_data[i]
                # Highlight best ask with red background
                is_best_ask = i == 0 and ask["price"] == best_ask
                if is_best_ask:
                    highlight_rect = mpatches.Rectangle(
                        (0.76, y_pos - 0.01),
                        0.18,
                        row_height,
                        transform=self.fig.transFigure,
                        facecolor="#440000",
                        edgecolor="#ff4444",
                        linewidth=1,
                    )
                    self.ax.add_patch(highlight_rect)

                self.ax.text(
                    0.78,
                    y_pos,
                    self._format_size(ask["cumulative"]),
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="#888888",
                    transform=self.fig.transFigure,
                )
                self.ax.text(
                    0.92,
                    y_pos,
                    self._format_size(ask["size"]),
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="#ff4444" if is_best_ask else "#aaaaaa",
                    fontweight="bold" if is_best_ask else "normal",
                    transform=self.fig.transFigure,
                )

        # Draw separator line between bids and asks
        separator_y = start_y - max_rows * row_height - 0.02
        from matplotlib.lines import Line2D

        separator_line = Line2D(
            [0.06, 0.94],
            [separator_y, separator_y],
            transform=self.fig.transFigure,
            color="#555555",
            linewidth=2,
        )
        self.ax.add_line(separator_line)

        # Mid price indicator
        self.ax.text(
            0.5,
            separator_y - 0.015,
            f"Mid: {self._format_price(mid_price)}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="#333333", edgecolor="#555555"
            ),
            transform=self.fig.transFigure,
        )

        # Footer info
        footer_y = 0.15
        self.ax.text(
            0.5,
            footer_y,
            f"State {self.current_index + 1} of {len(self.orderbook.getStates())} | "
            f"Use slider or arrow keys (←/→) to navigate",
            ha="center",
            va="top",
            fontsize=10,
            color="#888888",
            transform=self.fig.transFigure,
        )

        self.fig.canvas.draw()

    def _on_slider_change(self, val):
        """Handle slider change event."""
        self.current_index = int(val)
        self._update_display()

    def _prev_state(self, event):
        """Go to previous state."""
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.set_val(self.current_index)
            self._update_display()

    def _next_state(self, event):
        """Go to next state."""
        if self.current_index < len(self.orderbook.getStates()) - 1:
            self.current_index += 1
            self.slider.set_val(self.current_index)
            self._update_display()

    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == "left" or event.key == "up":
            self._prev_state(None)
        elif event.key == "right" or event.key == "down":
            self._next_state(None)

    def _show_chart(self, event):
        """Show the orderbook chart plot using orderbook.plot()."""
        import matplotlib.pyplot as plt

        # Call orderbook.plot() with show_bidask=True and max_level=0 to show best bid/ask
        # max_level=0 gets the best bid/ask (first in list), not max_level=-1 (last/worst)
        # This will open in a separate window
        try:
            self.orderbook.plot(show_bidask=True, max_level=0, show=True)
        except (IndexError, AttributeError) as e:
            # Fallback if there's an issue with bid/ask data (e.g., empty orderbook)
            print(f"Error showing chart with bid/ask: {e}")
            try:
                self.orderbook.plot(show_bidask=False, show=True)
            except Exception as e2:
                print(f"Error showing chart: {e2}")

    def show(self):
        """Display the orderbook UI."""
        self._create_orderbook_display()
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the orderbook UI."""
    import sys

    # Load orderbook
    orderbook = Orderbook()
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Loading orderbook from {file_path}...")
        try:
            orderbook.loadFromEvents(file_path)
            print(f"Loaded {len(orderbook.getStates())} states")
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Creating artificial orderbook instead...")
            import datetime

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
    else:
        # Default: try to load from standard location
        try:
            orderbook.loadFromEvents("data/events/ob-train.tsv")
            print(f"Loaded {len(orderbook.getStates())} states from ob-train.tsv")
        except:
            print("Creating artificial orderbook...")
            import datetime

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

    # Create and show UI
    ui = OrderbookUI(orderbook)
    ui.show()


if __name__ == "__main__":
    main()
