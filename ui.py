import matplotlib.pyplot as plt
import matplotlib.animation as animation


class UI:

    @staticmethod
    def animate(f, interval=5000, axis=[0, 100, -50, 50], frames=None):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.axis(axis)
        ax1.autoscale(True)
        xs = []
        ys = []

        def do_animate(i, f, ax1, xs, ys):
            y = f()
            if len(xs) == 0:
                xs.append(0)
            else:
                xs.append(xs[-1] + 1)
            ys.append(y)
            ax1.clear()
            ax1.plot(xs, ys)

        ani = animation.FuncAnimation(
            fig,
            lambda i: do_animate(i, f, ax1, xs, ys),
            interval=interval,
            frames=frames
        )
        # from IPython.display import HTML
        # HTML(ani.to_jshtml())
        plt.show()
