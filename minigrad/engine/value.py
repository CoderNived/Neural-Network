# engine/value.py

import numpy as np


class Value:
    def __init__(self, data, _parents=(), _op='', _label=''):

        self._data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self._data)

        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op
        self._label = _label

    # ==========================================================
    # Data property (prevents float assignment corruption)
    # ==========================================================

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.array(value, dtype=float)

        if hasattr(self, "grad"):
            if self.grad.shape != self._data.shape:
                self.grad = np.zeros_like(self._data)

    # ==========================================================
    # Utility
    # ==========================================================

    def __repr__(self):
        return (
            f"Value(data={self.data}, "
            f"grad={self.grad}, "
            f"op='{self._op}')"
        )

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    @property
    def is_scalar(self):
        return self.data.shape == ()

    # ==========================================================
    # Arithmetic Operations
    # ==========================================================

    def __add__(self, other):

        other = Value._wrap(other)

        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                out.grad,
                self.data.shape
            )

            other.grad += _reduce_to_shape(
                out.grad,
                other.data.shape
            )

        out._backward = _backward
        return out

    def __mul__(self, other):

        other = Value._wrap(other)

        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                other.data * out.grad,
                self.data.shape
            )

            other.grad += _reduce_to_shape(
                self.data * out.grad,
                other.data.shape
            )

        out._backward = _backward
        return out

    def __pow__(self, exponent):

        assert isinstance(exponent, (int, float))

        out = Value(
            self.data ** exponent,
            (self,),
            f'**{exponent}'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                exponent *
                (self.data ** (exponent - 1)) *
                out.grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (Value._wrap(other) ** (-1))

    def __rtruediv__(self, other):
        return Value._wrap(other) * (self ** (-1))

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value._wrap(other))

    def __rsub__(self, other):
        return Value._wrap(other) + (-self)

    __radd__ = __add__
    __rmul__ = __mul__

    # ==========================================================
    # Activations
    # ==========================================================

    def relu(self):

        out = Value(
            np.maximum(0, self.data),
            (self,),
            'relu'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                (self.data > 0).astype(float) * out.grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    def sigmoid(self):

        s = 1 / (1 + np.exp(-self.data))

        out = Value(
            s,
            (self,),
            'sigmoid'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                s * (1 - s) * out.grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    def tanh(self):

        t = np.tanh(self.data)

        out = Value(
            t,
            (self,),
            'tanh'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                (1 - t**2) * out.grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    def exp(self):

        e = np.exp(self.data)

        out = Value(
            e,
            (self,),
            'exp'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                e * out.grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    def log(self):

        if np.any(self.data <= 0):
            raise ValueError(
                "log input must be positive"
            )

        out = Value(
            np.log(self.data),
            (self,),
            'log'
        )

        def _backward():

            self.grad += _reduce_to_shape(
                (1 / self.data) * out.grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    # ==========================================================
    # Reductions
    # ==========================================================

    def mean(self):

        n = self.data.size

        out = Value(
            self.data.mean(),
            (self,),
            'mean'
        )

        def _backward():

            self.grad += (
                np.ones_like(self.data)
                * out.grad / n
            )

        out._backward = _backward
        return out

    def sum(self):

        out = Value(
            self.data.sum(),
            (self,),
            'sum'
        )

        def _backward():

            self.grad += (
                np.ones_like(self.data)
                * out.grad
            )

        out._backward = _backward
        return out

    # ==========================================================
    # Backpropagation
    # ==========================================================

    def backward(self):

        topo = []
        visited = set()

        def build(v):

            if id(v) not in visited:

                visited.add(id(v))

                for p in v._parents:
                    build(p)

                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):

        visited = set()

        def zero(v):

            if id(v) not in visited:

                visited.add(id(v))

                v.grad = np.zeros_like(v.data)

                for p in v._parents:
                    zero(p)

        zero(self)


# ==========================================================
# Helper
# ==========================================================

def _reduce_to_shape(
    grad,
    target_shape
):

    grad = np.array(
        grad,
        dtype=float
    )

    if target_shape == ():
        return grad.sum()

    while grad.ndim > len(target_shape):

        grad = grad.sum(
            axis=0
        )

    for i, (g, t) in enumerate(
        zip(
            grad.shape,
            target_shape
        )
    ):

        if t == 1 and g > 1:

            grad = grad.sum(
                axis=i,
                keepdims=True
            )

    return grad