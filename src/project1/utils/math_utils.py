"""Utility math functions for Project 1."""

import numpy as np


def sigmoid(x):
    """Compute the logistic sigmoid element-wise for the input array."""
    return 1 / (1 + np.exp(-x))


__all__ = ["sigmoid"]
