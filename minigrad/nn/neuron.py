from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from engine.value import Value
from engine.ops import ACTIVATIONS, get_activation


# ── weight initialisers ───────────────────────────────────────────────────────

def _xavier_uniform(n_inputs: int, n_outputs: int = 1) -> float:
    """Glorot uniform: keeps variance stable across layers for tanh/sigmoid."""
    limit = math.sqrt(6.0 / (n_inputs + n_outputs))
    return random.uniform(-limit, limit)


def _he_uniform(n_inputs: int) -> float:
    """He uniform: correct for ReLU-family (accounts for dead-half of ReLU)."""
    limit = math.sqrt(6.0 / n_inputs)
    return random.uniform(-limit, limit)


# Map activation name → best-practice initialiser
_INIT_STRATEGY: dict[str, Callable[[int], float]] = {
    'relu':       _he_uniform,
    'leaky_relu': _he_uniform,
    'elu':        _he_uniform,
    'swish':      _he_uniform,
    'sigmoid':    _xavier_uniform,
    'tanh':       _xavier_uniform,
    'linear':     _xavier_uniform,
}


# ── Neuron ────────────────────────────────────────────────────────────────────

class Neuron:
    """
    A single artificial neuron with persistent weights and a configurable
    activation function.

    Forward pass:  out = activation(w · x + b)
    Backward pass: handled by the Value computation graph (call .backward()
                   on the loss, then read .grad on each parameter).
    """

    def __init__(
        self,
        n_inputs:   int,
        activation: str  = 'tanh',
        *,
        bias:       bool = True,
    ) -> None:
        """
        Args:
            n_inputs:   Number of input connections (fan-in).
            activation: Name of the activation function (see ACTIVATIONS).
            bias:       Whether to include a learnable bias term.
        """
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be ≥ 1, got {n_inputs}.")

        self._activation_name = activation.lower()
        self.activation       = get_activation(self._activation_name)

        # Pick the initialiser that matches the activation
        init = _INIT_STRATEGY.get(self._activation_name, _xavier_uniform)
        self.w = [Value(init(n_inputs)) for _ in range(n_inputs)]
        self.b = Value(0.0) if bias else None   # bias at 0: no privileged direction

    # ── forward ──────────────────────────────────────────────────────────────

    def __call__(self, x: Sequence[float | Value]) -> Value:
        """
        Compute activation(w · x + b).

        Args:
            x: Inputs — plain floats or Value nodes (mixed is fine).

        Returns:
            A Value node that is the root of the new sub-graph for this call.
            Weights are the same persistent objects across calls.
        """
        if len(x) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} inputs, got {len(x)}."
            )

        # Coerce once; keep graph nodes uniform
        inputs = [xi if isinstance(xi, Value) else Value(float(xi)) for xi in x]

        # Fused dot-product: z = b + Σ wᵢxᵢ
        # Starting from bias (or 0) avoids an extra addition node
        z: Value = self.b if self.b is not None else Value(0.0)
        for wi, xi in zip(self.w, inputs):
            z = z + wi * xi

        return self.activation(z)

    # ── parameter access ─────────────────────────────────────────────────────

    def parameters(self) -> list[Value]:
        """
        All trainable parameters as a flat list of the *original* Value objects.
        Optimisers and gradient clippers must receive these exact references.
        """
        params = list(self.w)
        if self.b is not None:
            params.append(self.b)
        return params

    def zero_grad(self) -> None:
        """Reset accumulated gradients on every parameter to 0.0."""
        for p in self.parameters():
            p.grad = 0.0

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def fan_in(self) -> int:
        return len(self.w)

    @property
    def n_params(self) -> int:
        return len(self.parameters())

    def __repr__(self) -> str:
        bias_str = f", bias={self.b.data:.4f}" if self.b is not None else ", bias=off"
        return (
            f"Neuron(fan_in={self.fan_in}, "
            f"activation={self._activation_name!r}"
            f"{bias_str})"
        )