"""
engine/activations.py

Activation functions that operate on Value objects.
Each function is also available as a method on Value itself
(e.g. x.softmax()), but standalone versions live here for
use in model definitions without importing Value directly.
"""

import numpy as np
from engine.value import Value


def relu(x):
    """Element-wise ReLU."""
    return Value._wrap(x).relu()


def sigmoid(x):
    """Element-wise sigmoid."""
    return Value._wrap(x).sigmoid()


def tanh(x):
    """Element-wise tanh."""
    return Value._wrap(x).tanh()


def softmax(x, axis=-1):
    """
    Numerically stable softmax along `axis`.

    Parameters
    ----------
    x    : Value or array-like, shape (..., C, ...)
    axis : int  Axis along which to compute probabilities (default: -1)

    Returns
    -------
    Value  Same shape as x, values in (0, 1) summing to 1 along `axis`.

    Example
    -------
    >>> logits = Value([[2.0, 1.0, 0.1]])
    >>> probs  = softmax(logits)          # shape (1, 3)
    >>> probs.data.sum(axis=-1)           # [1.]
    """
    return Value._wrap(x).softmax(axis=axis)