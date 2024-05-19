import numpy as np


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# tanh function
def tanh(x):
    return np.tanh(x)


# RELU function
def relu(x):
    return np.maximum(0, x)


# Leaky RELU function
def leaky_relu(x):
    return np.maximum(0.01 * x, x)


# plot leaky relu function using matplotlib
def plot_leaky_relu():
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10, 1000)
    y = leaky_relu(x)
    plt.plot(x, y)
    plt.show()


plot_leaky_relu()
