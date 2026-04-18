import math
from engine.value import Value

def relu(v):
    v = v if isinstance(v, Value) else Value(v)
    out = Value(max(0.0, v.data), _parents=(v,), _op='relu')
    def _backward():
        v.grad += (1.0 if v.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out

def sigmoid(v):
    v = v if isinstance(v, Value) else Value(v)
    # numerically stable form (same logic as Phase 1)
    x = v.data
    if x >= 0:
        s = 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        s = ex / (1.0 + ex)
    out = Value(s, _parents=(v,), _op='sigmoid')
    def _backward():
        # d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
        v.grad += s * (1.0 - s) * out.grad
    out._backward = _backward
    return out

def tanh(v):
    v = v if isinstance(v, Value) else Value(v)
    x = v.data
    if x >= 0:
        u = math.exp(-2 * x)
        t = (1.0 - u) / (1.0 + u)
    else:
        u = math.exp(2 * x)
        t = (u - 1.0) / (u + 1.0)
    out = Value(t, _parents=(v,), _op='tanh')
    def _backward():
        # d(tanh)/dx = 1 - tanh²(x)
        v.grad += (1.0 - t ** 2) * out.grad
    out._backward = _backward
    return out

ACTIVATIONS = {
    'relu':    relu,
    'sigmoid': sigmoid,
    'tanh':    tanh,
    'linear':  lambda v: v,   # no-op, used for output neurons
}