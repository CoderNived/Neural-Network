"""
nn/linear.py

Fully-connected (affine) layer: y = x @ W + b
"""

import numpy as np
from engine.value import Value


class Linear:
    """
    Fully-connected layer.

    Parameters
    ----------
    in_features  : int  Number of input features.
    out_features : int  Number of output features.
    bias         : bool Include a bias term (default: True).

    Attributes
    ----------
    W : Value, shape (in_features, out_features)
    b : Value, shape (1, out_features)   – broadcasts over batch

    Example
    -------
    >>> layer = Linear(3, 4)
    >>> x     = Value(np.random.randn(8, 3))   # batch of 8
    >>> y     = layer(x)                        # shape (8, 4)
    >>> loss  = y.sum()
    >>> loss.backward()
    >>> layer.W.grad                            # shape (3, 4)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features

        # Kaiming / He initialisation — good default for ReLU networks
        std = np.sqrt(2.0 / in_features)
        self.W = Value(
            np.random.randn(in_features, out_features) * std
        )

        self.b = Value(np.zeros((1, out_features))) if bias else None

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : Value, shape (N, in_features)

        Returns
        -------
        Value, shape (N, out_features)
        """
        x   = Value._wrap(x)
        out = x @ self.W
        if self.b is not None:
            out = out + self.b      # bias broadcasts: (N, C) + (1, C)
        return out

    # ----------------------------------------------------------
    # Parameter access
    # ----------------------------------------------------------

    def parameters(self):
        """Return all trainable parameters as a list of Value objects."""
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def zero_grad(self):
        """Zero gradients of all parameters."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    # ----------------------------------------------------------
    # Repr
    # ----------------------------------------------------------

    def __repr__(self):
        bias_str = f", bias={self.b is not None}"
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}"
            f"{bias_str})"
        )