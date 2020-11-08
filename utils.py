import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_bayes_beta(shape_prior, shape_obs, fig_size=(10, 10)):
    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=fig_size)

    a_prior, b_prior = shape_prior
    a_obs, b_obs = shape_obs
    a_post, b_post = a_prior+a_obs, b_prior+b_obs

    ax.plot(x, stats.beta.pdf(x, a_prior, b_prior), label='prior', linestyle=':', color='g')
    ax.plot(x, stats.beta.pdf(x, a_obs, b_obs), label='likelihood', linestyle='-', color='b')
    ax.plot(x, stats.beta.pdf(x, a_post, b_post), label='posterior', linestyle='--', color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('pdf')
    ax.legend()

    plt.show()


def plot_nominal(x_labels, y_labels, fig_size=(10, 10), tick_size=10, annots=[], show_direction=True):
    """
    Plot ordinal / nominal entities in an coordinate system.

    params:
        x_labels:
        y_labels:
        fig_size:
        tick_size:
        annots:
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_direction:
        plt.plot(range(len(x_labels)+1), range(len(y_labels)+1), 'r')
        plt.arrow(0.5, 0.5, 0.1, 0.1, width=0.05, color='r')

    ax.set_xticks(list(range(1, len(x_labels)+1)))
    ax.set_yticks(list(range(1, len(y_labels)+1)))
    ax.set_xticklabels(x_labels, fontsize=tick_size)
    ax.set_yticklabels(y_labels, fontsize=tick_size)

    for (x,y), annot_string in annots:
        ax.annotate(annot_string, (x,y), fontsize=tick_size)

    plt.show()

