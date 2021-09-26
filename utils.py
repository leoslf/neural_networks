import numpy as np
import matplotlib.pyplot as plt

def to_column_vector(y):
    return np.reshape(y, (1, 2)).T

def array_partition(X, mask):
    return X[mask], X[~mask]

def plot_scatter(X, marker = None, color = None):
    plt.scatter(X[:, 0], X[:, 1], marker = marker, color = color)

def plot_line(classifier):
    w = classifier.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (classifier.intercept_[0] / w[1])

    plt.plot(xx, yy, "k-")

def plot_decision_boundary(classifier, X, y_true):
    """ Decision boundary plot for 2D data points with binary labels """
    X_0, X_1 = array_partition(X, y_true == 0)

    fig = plt.figure()
    plot_scatter(X_0, marker = "o")
    plot_scatter(X_1, marker = "+", color = "green")

    plot_line(classifier)

    return fig
