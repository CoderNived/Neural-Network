# engine/activations.py

import numpy as np
from engine.value import Value


# ==========================================================
# Helper
# ==========================================================

def _ensure_value(v):
    """Convert scalars/lists to Value."""
    return v if isinstance(v, Value) else Value(v)


# ==========================================================
# Activation Functions
# ==========================================================

def relu(v):
    """
    ReLU:
        f(x)=max(0,x)
    """
    v = _ensure_value(v)

    out = Value(
        np.maximum(0.0, v.data),
        _parents=(v,),
        _op='relu'
    )

    def _backward():
        v.grad += (
            (v.data > 0).astype(float)
            * out.grad
        )

    out._backward = _backward
    return out


def sigmoid(v):
    """
    Numerically stable sigmoid:
        σ(x)=1/(1+e^(-x))
    """

    v = _ensure_value(v)

    x = v.data

    s = np.where(
        x >= 0,
        1.0/(1.0+np.exp(-x)),
        np.exp(x)/(1.0+np.exp(x))
    )

    out = Value(
        s,
        _parents=(v,),
        _op='sigmoid'
    )

    def _backward():

        v.grad += (
            s*(1.0-s)
            * out.grad
        )

    out._backward = _backward
    return out


def tanh(v):
    """
    tanh activation
    """

    v = _ensure_value(v)

    t = np.tanh(v.data)

    out = Value(
        t,
        _parents=(v,),
        _op='tanh'
    )

    def _backward():

        v.grad += (
            (1.0 - t**2)
            * out.grad
        )

    out._backward = _backward
    return out


def leaky_relu(v, negative_slope=0.01):
    """
    Leaky ReLU:
        x if x>0
        αx otherwise
    """

    v = _ensure_value(v)

    out_data = np.where(
        v.data > 0,
        v.data,
        negative_slope*v.data
    )

    out = Value(
        out_data,
        _parents=(v,),
        _op='leaky_relu'
    )

    def _backward():

        grad = np.where(
            v.data > 0,
            1.0,
            negative_slope
        )

        v.grad += (
            grad
            * out.grad
        )

    out._backward = _backward
    return out


def elu(v, alpha=1.0):
    """
    ELU:
        x if x>0
        α(exp(x)-1) otherwise
    """

    v = _ensure_value(v)

    out_data=np.where(
        v.data>0,
        v.data,
        alpha*(np.exp(v.data)-1)
    )

    out=Value(
        out_data,
        _parents=(v,),
        _op='elu'
    )

    def _backward():

        grad=np.where(
            v.data>0,
            1.0,
            out.data+alpha
        )

        v.grad += (
            grad
            * out.grad
        )

    out._backward=_backward
    return out


def swish(v):
    """
    Swish / SiLU:
        x*σ(x)
    """

    v = _ensure_value(v)

    x=v.data

    s=np.where(
        x>=0,
        1/(1+np.exp(-x)),
        np.exp(x)/(1+np.exp(x))
    )

    out=Value(
        x*s,
        _parents=(v,),
        _op='swish'
    )

    def _backward():

        grad=(
            s+
            x*s*(1-s)
        )

        v.grad += (
            grad
            * out.grad
        )

    out._backward=_backward
    return out


def linear(v):
    """
    Identity function
    """

    return _ensure_value(v)


# ==========================================================
# Registry
# ==========================================================

ACTIVATIONS = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "swish": swish,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "linear": linear,
}


def get_activation(name):

    name = name.lower()

    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(ACTIVATIONS.keys())}"
        )

    return ACTIVATIONS[name]

# just to make the commit 