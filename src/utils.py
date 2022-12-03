import matplotlib as plt
import numpy as np
from matplotlib import pyplot as plt

class RLAlgo:

    def learn(self, environment, **kwargs):
        pass

    def choose_action(self, observation):
        pass

def display_plot_with_variance(x, y, std, x_label, y_label,):
    plt.plot(x, y, 'k-')
    plt.fill_between(x, y-std, y+std)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()