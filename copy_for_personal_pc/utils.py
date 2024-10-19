import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(indeces, n, B):
    """Function to use for interactive plotting.
    interact(lambda n: plot_histogram(something.K_RESAMPLED[1:], n, B=6) , n=(0, len(something.εs) - 2))
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(start=-0.5, stop=(B+1), step=1)
    _ = ax.hist(indeces[n, :], bins=bins, edgecolor='k', color='lightsalmon')
    ax.set_xticks(np.arange(B+1))
    return plt.show()


def plot_histogram_safe(indeces_n, B, figsize=None, xticks=None):
    """This is the same as above but used when using NHUG.
    """
    figsize = (10, 4) if figsize is None else figsize
    xticks = np.arange(B+1) if xticks is None else xticks
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(start=-0.5, stop=(B+1), step=1)
    _ = ax.hist(indeces_n, bins=bins, edgecolor='k', color='lightsalmon')
    ax.set_xticks(xticks)
    return plt.show()