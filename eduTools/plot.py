import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def play_movie(data, fps=60, fig=None, ax=None, outpath=None):
    """ plots scatterplots at a framerate. 

    
    Parameters
    ----------
    data : Array | (n_frames, n_coordinates, 2)
        data to plot. First dimension represents the number of frames
        to play. Second dimension the number of (x,y) coordinates per
        frame. Third dimension the x and y coordinate respectively.
    fps : Integer | Float
        Framerate per second
    fig : Figure instance
        Figure where the clip is being added to
    ax : Axes instance
        Axes object on which to draw the scatter plot
    outpath : String | Path
        Location where the figure should be drawn to
    """

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    elif fig is None or ax is None:
        raise ValueError("Provide both ax and fig or none of them.")

    #  plot
    x = data[0, :, 0]
    y = data[0, :, 1]
    setup, = ax.plot(x, y, '.')
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, outpath, dpi=300):
        for frame in range(1, data.shape[0]):
            x = data[frame, :, 0]
            y = data[frame, :, 1]
            setup.set_xdata(x)
            setup.set_ydata(y)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            writer.grab_frame()


def save_plot(fig, outpath, **kwargs):
    """ Convenience wrapper for saving and closing plots."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, **kwargs)
    fig.clf()
    plt.close()