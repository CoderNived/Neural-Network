import math
from engine.value import Value

# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_value(v) -> Value:
    """Coerce scalars to Value; pass Value instances through unchanged."""
    return v if isinstance(v, Value) else Value(float(v))


# ── activation functions ──────────────────────────────────────────────────────

def relu(v: Value) -> Value:
    v = _ensure_value(v)
    data = max(0.0, v.data)
    out  = Value(data, _parents=(v,), _op='relu')

    def _backward():
        v.grad += (out.grad if v.data > 0 else 0.0)

    out._backward = _backward
    return out


def sigmoid(v: Value) -> Value:
    """Numerically stable sigmoid via log-sum-exp trick."""
    v = _ensure_value(v)
    x = v.data
    # Avoid overflow in exp by choosing the stable branch
    s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    out = Value(s, _parents=(v,), _op='sigmoid')

    def _backward():
        v.grad += s * (1.0 - s) * out.grad   # σ(1 − σ)

    out._backward = _backward
    return out


def tanh(v: Value) -> Value:
    """Numerically stable tanh (avoids catastrophic cancellation for large |x|)."""
    v = _ensure_value(v)
    x = v.data
    # math.tanh already handles stability internally; use it directly
    t = math.tanh(x)
    out = Value(t, _parents=(v,), _op='tanh')

    def _backward():
        v.grad += (1.0 - t * t) * out.grad   # sech²(x)

    out._backward = _backward
    return out


def leaky_relu(v: Value, negative_slope: float = 0.01) -> Value:
    """Leaky ReLU — keeps a small gradient for negative inputs."""
    v = _ensure_value(v)
    data = v.data if v.data > 0 else negative_slope * v.data
    out  = Value(data, _parents=(v,), _op='leaky_relu')

    def _backward():
        v.grad += (out.grad if v.data > 0 else negative_slope * out.grad)

    out._backward = _backward
    return out


def elu(v: Value, alpha: float = 1.0) -> Value:
    """Exponential Linear Unit — smooth negative region, zero-centered outputs."""
    v = _ensure_value(v)
    data = v.data if v.data > 0 else alpha * (math.exp(v.data) - 1.0)
    out  = Value(data, _parents=(v,), _op='elu')

    def _backward():
        v.grad += out.grad if v.data > 0 else (out.data + alpha) * out.grad

    out._backward = _backward
    return out


def swish(v: Value) -> Value:
    """Swish / SiLU: x · σ(x). Smooth, non-monotonic, often beats ReLU."""
    v = _ensure_value(v)
    x = v.data
    s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    out = Value(x * s, _parents=(v,), _op='swish')

    def _backward():
        # d/dx [x·σ(x)] = σ(x) + x·σ(x)·(1 − σ(x))
        v.grad += (s + x * s * (1.0 - s)) * out.grad

    out._backward = _backward
    return out


def linear(v: Value) -> Value:
    """Identity / no-op — used for output neurons (regression, etc.)."""
    return _ensure_value(v)


# ── registry ──────────────────────────────────────────────────────────────────

ACTIVATIONS: dict[str, callable] = {
    'relu':       relu,
    'leaky_relu': leaky_relu,
    'elu':        elu,
    'swish':      swish,
    'sigmoid':    sigmoid,
    'tanh':       tanh,
    'linear':     linear,
}


def get_activation(name: str) -> callable:
    """Look up an activation by name with a clear error on miss."""
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {sorted(ACTIVATIONS)}"
        )