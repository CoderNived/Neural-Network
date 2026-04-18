import random
import math
from engine.value import Value
from engine.ops import ACTIVATIONS

class Neuron:
    def __init__(self, n_inputs, activation='tanh'):
        # Xavier initialization: std = 1/sqrt(n_inputs)
        # Keeps Var(z) = 1 regardless of fan-in.
        # Prevents saturation at init, ensures nonzero gradients.
        scale = 1.0 / math.sqrt(n_inputs)
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)   # bias at zero: no privileged direction at init

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(ACTIVATIONS.keys())}"
            )
        self.activation = ACTIVATIONS[activation]
        self._activation_name = activation

    def __call__(self, x):
        """
        x: list of floats or Value objects.
        Builds a computation graph:  activation(w·x + b)
        The graph is new each call; weights are the same persistent objects.
        """
        if len(x) != len(self.w):
            raise ValueError(
                f"Input length {len(x)} doesn't match "
                f"neuron fan-in {len(self.w)}"
            )
        # Wrap plain floats so the graph is uniform
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]

        # Weighted sum — each term is a graph node
        z = self.b
        for wi, xi in zip(self.w, x):
            z = z + wi * xi

        return self.activation(z)

    def parameters(self):
        """
        Returns the original Value objects — not copies.
        The optimizer must receive these exact objects to update real weights.
        """
        return self.w + [self.b]

    def zero_grad(self):
        """
        Resets gradients on all parameters.
        Lives here (not on Value) because zero_grad is a training-loop
        concern, not an autodiff concern. Value just accumulates.
        """
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return (f"Neuron(fan_in={len(self.w)}, "
                f"activation={self._activation_name})")