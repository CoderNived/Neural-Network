"""
nn/network.py — Top-level trainable model for minigrad.

Class hierarchy
───────────────
    Module              Base class every trainable object inherits from.
                        Provides: parameters(), zero_grad(), train(), eval(),
                                  state_dict(), load_state_dict().

    Sequential(Module)  Ordered container of named Layers.
                        Chains their forward passes automatically.

    Network(Sequential) Full model with summary(), save(), load(),
                        clip_grad_norm(), and a built-in training step.

Typical usage
─────────────
    from nn.network import Network
    from nn.layer   import Layer

    model = Network([
        Layer(2,  16, 'tanh'),
        Layer(16,  8, 'tanh'),
        Layer(8,   1, 'linear'),
    ], name='XOR-net')

    print(model.summary())

    # Training loop
    for epoch in range(100):
        loss = compute_loss(model, xs, ys)
        model.zero_grad()
        loss.backward()
        model.clip_grad_norm(max_norm=1.0)   # optional
        for p in model.parameters():
            p.data -= 0.01 * p.grad

    model.save('xor.json')

    # Later
    model2 = Network.load('xor.json')
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing      import Iterator, Sequence

from engine.value import Value
from nn.layer     import Layer


# ══════════════════════════════════════════════════════════════════════════════
# Module — base class
# ══════════════════════════════════════════════════════════════════════════════

class Module:
    """
    Base class for every trainable object in minigrad.

    Mirrors the PyTorch nn.Module interface at a minigrad scale so the
    transition to the real thing feels natural later.

    Subclasses must implement:
        forward(x)  — the actual computation
        parameters() — flat list of all trainable Value objects
    """

    # ── mode ─────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._training: bool = True

    @property
    def training(self) -> bool:
        """True while in training mode, False in eval/inference mode."""
        return self._training

    def train(self) -> Module:
        """Switch to training mode (enables dropout, batch-norm updates, etc.)."""
        self._training = True
        return self

    def eval(self) -> Module:
        """
        Switch to inference mode.
        Dropout becomes a no-op; batch-norm uses running statistics.
        """
        self._training = False
        return self

    # ── interface every subclass must fulfil ─────────────────────────────────

    def forward(self, x: Sequence[float | Value]) -> Value | list[Value]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()."
        )

    def parameters(self) -> list[Value]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement parameters()."
        )

    # ── helpers every subclass gets for free ─────────────────────────────────

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        return self.forward(x)

    def zero_grad(self) -> None:
        """Reset every parameter gradient to 0.0."""
        for p in self.parameters():
            p.grad = 0.0

    @property
    def n_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return len(self.parameters())

    # ── state dict ────────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, float]:
        """
        Snapshot of every parameter value keyed by its index.
        Used for saving / loading checkpoints.

        Returns:
            {'0': 0.123, '1': -0.456, ...}
        """
        return {str(i): p.data for i, p in enumerate(self.parameters())}

    def load_state_dict(self, state: dict[str, float]) -> None:
        """
        Restore parameter values from a state dict produced by state_dict().

        Args:
            state: Dict mapping str(index) → float value.

        Raises:
            ValueError: If the number of parameters doesn't match.
        """
        params = self.parameters()
        if len(state) != len(params):
            raise ValueError(
                f"State dict has {len(state)} entries but model has "
                f"{len(params)} parameters."
            )
        for i, p in enumerate(params):
            p.data = float(state[str(i)])

    # ── gradient utilities ────────────────────────────────────────────────────

    def clip_grad_norm(self, max_norm: float = 1.0) -> float:
        """
        Clip all parameter gradients by global L2 norm (in-place).

        Prevents exploding gradients in deep networks or early training.

        Args:
            max_norm: Clip threshold. Gradients are rescaled if their global
                      L2 norm exceeds this value.

        Returns:
            The unclipped global gradient norm (useful for monitoring).
        """
        params     = self.parameters()
        total_norm = math.sqrt(sum(p.grad ** 2 for p in params))

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-12)   # +eps avoids /0
            for p in params:
                p.grad *= scale

        return total_norm

    def clip_grad_value(self, clip_value: float = 1.0) -> None:
        """
        Clip each gradient independently to [-clip_value, +clip_value].

        Simpler than norm clipping; less aware of gradient direction.

        Args:
            clip_value: Absolute maximum allowed gradient magnitude.
        """
        for p in self.parameters():
            p.grad = max(-clip_value, min(clip_value, p.grad))


# ══════════════════════════════════════════════════════════════════════════════
# Sequential — ordered container of named layers
# ══════════════════════════════════════════════════════════════════════════════

class Sequential(Module):
    """
    Chains a list of Layer objects: output of layer[i] feeds into layer[i+1].

    Layers can be accessed by index or by name:
        model[0]          first layer
        model['hidden1']  layer named 'hidden1'

    Args:
        layers:  Ordered list of Layer objects.
        names:   Optional list of string names (same length as layers).
                 Auto-names are 'layer_0', 'layer_1', … if omitted.
    """

    def __init__(
        self,
        layers: list[Layer],
        names:  list[str] | None = None,
    ) -> None:
        super().__init__()

        if not layers:
            raise ValueError("Sequential requires at least one Layer.")

        if names is not None and len(names) != len(layers):
            raise ValueError(
                f"Got {len(layers)} layers but {len(names)} names."
            )

        auto_names  = names or [f"layer_{i}" for i in range(len(layers))]
        seen: set[str] = set()
        for name in auto_names:
            if name in seen:
                raise ValueError(f"Duplicate layer name: '{name}'.")
            seen.add(name)

        # OrderedDict preserves insertion order and allows name-based lookup
        self._layers: OrderedDict[str, Layer] = OrderedDict(
            zip(auto_names, layers)
        )

    # ── layer access ──────────────────────────────────────────────────────────

    @property
    def layers(self) -> list[Layer]:
        """Ordered list of Layer objects."""
        return list(self._layers.values())

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers.keys())

    def __getitem__(self, key: int | str) -> Layer:
        if isinstance(key, int):
            return self.layers[key]
        return self._layers[key]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers.values())

    def add(self, layer: Layer, name: str | None = None) -> Sequential:
        """
        Append a layer to the end of the sequence.

        Args:
            layer: The Layer to add.
            name:  Optional name. Auto-assigned as 'layer_N' if omitted.

        Returns:
            self — allows chaining:  model.add(l1).add(l2)
        """
        name = name or f"layer_{len(self._layers)}"
        if name in self._layers:
            raise ValueError(f"Layer name '{name}' already exists.")
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected a Layer, got {type(layer).__name__}.")
        self._layers[name] = layer
        return self

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Sequence[float | Value]) -> Value | list[Value]:
        """
        Thread x through every layer in registration order.

        Each layer's output is wrapped in a list (if it isn't already)
        before being passed to the next layer so the interface stays uniform.
        """
        out: list[Value] | Value = list(x)
        for layer in self._layers.values():
            out = layer(out)
            if isinstance(out, Value):
                out = [out]

        return out[0] if len(out) == 1 else out

    # ── parameters ────────────────────────────────────────────────────────────

    def parameters(self) -> list[Value]:
        return [p for layer in self._layers.values() for p in layer.parameters()]


# ══════════════════════════════════════════════════════════════════════════════
# Network — full model with I/O and training utilities
# ══════════════════════════════════════════════════════════════════════════════

class Network(Sequential):
    """
    A named, serialisable neural network built from an ordered list of Layers.

    Extends Sequential with:
        • model.summary()       — pretty-printed architecture table
        • model.save(path)      — serialise weights + architecture to JSON
        • Network.load(path)    — reconstruct model from JSON checkpoint
        • model.grad_norm()     — inspect current gradient health
        • model.weight_stats()  — per-layer weight mean / std for debugging

    Args:
        layers: Ordered list of Layer objects that form the network.
        names:  Optional per-layer names shown in summary and checkpoint.
        name:   Human-readable name for the whole model (default 'Network').

    Example:
        model = Network([
            Layer(2,  16, 'tanh'),
            Layer(16,  8, 'tanh'),
            Layer(8,   1, 'linear'),
        ], name='XOR-net')
    """

    def __init__(
        self,
        layers: list[Layer],
        names:  list[str] | None = None,
        *,
        name:   str = 'Network',
    ) -> None:
        super().__init__(layers, names)
        self.name = name

    # ── validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_layers(layers: list[Layer]) -> None:
        """
        Check that consecutive layers have matching dimensions.
        Catches shape mismatches at construction time, not at runtime.
        """
        for i in range(len(layers) - 1):
            out_size = layers[i].n_outputs
            in_size  = layers[i + 1].n_inputs
            if out_size != in_size:
                raise ValueError(
                    f"Shape mismatch between layer {i} (out={out_size}) "
                    f"and layer {i + 1} (in={in_size})."
                )

    # ── override __init__ to add validation ───────────────────────────────────

    def __new__(cls, layers, names=None, *, name='Network'):
        # Validate before Sequential.__init__ allocates anything
        Network._validate_layers(layers)
        return super().__new__(cls)

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Print a Keras-style architecture table.

            ════════════════════════════════════════════════════════════
                                    XOR-net
            ════════════════════════════════════════════════════════════
            #   Name        Shape       Activation     Params
            ────────────────────────────────────────────────────────────
            0   layer_0     2  →  16    tanh               48
            1   layer_1     16 →   8    tanh              136
            2   layer_2     8  →   1    linear              9
            ────────────────────────────────────────────────────────────
            Total params                                   193
            Trainable                                      193
            Mode                                         train
            ════════════════════════════════════════════════════════════
        """
        W = 60
        lines = [
            "═" * W,
            f"{self.name:^{W}}",
            "═" * W,
            f"{'#':<4} {'Name':<12} {'Shape':<14} {'Activation':<14} {'Params':>6}",
            "─" * W,
        ]
        for i, (layer_name, layer) in enumerate(self._layers.items()):
            shape = f"{layer.n_inputs:<4}→ {layer.n_outputs}"
            lines.append(
                f"{i:<4} {layer_name:<12} {shape:<14} "
                f"{layer._activation_name:<14} {layer.n_params:>6}"
            )
        lines += [
            "─" * W,
            f"{'Total params':<48} {self.n_params:>6}",
            f"{'Trainable':<48} {self.n_params:>6}",
            f"{'Mode':<48} {'train' if self._training else 'eval':>6}",
            "═" * W,
        ]
        result = "\n".join(lines)
        print(result)
        return result

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the model architecture + weights to a JSON checkpoint.

        Checkpoint schema:
        {
          "name":   "XOR-net",
          "layers": [
            { "name": "layer_0", "n_inputs": 2,  "n_outputs": 16,
              "activation": "tanh", "bias": true },
            ...
          ],
          "state_dict": { "0": 0.123, "1": -0.456, ... }
        }

        Args:
            path: File path. Directory is created automatically if missing.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        checkpoint = {
            "name":   self.name,
            "layers": [
                {
                    "name":       layer_name,
                    "n_inputs":   layer.n_inputs,
                    "n_outputs":  layer.n_outputs,
                    "activation": layer._activation_name,
                    "bias":       layer.neurons[0].b is not None,
                }
                for layer_name, layer in self._layers.items()
            ],
            "state_dict": self.state_dict(),
        }
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Network:
        """
        Reconstruct a Network from a JSON checkpoint produced by save().

        Args:
            path: Path to the .json checkpoint file.

        Returns:
            A fully restored Network with all weights loaded.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: '{path}'")

        with open(path) as f:
            checkpoint = json.load(f)

        layers = [
            Layer(
                n_inputs   = ld["n_inputs"],
                n_outputs  = ld["n_outputs"],
                activation = ld["activation"],
                bias       = ld["bias"],
            )
            for ld in checkpoint["layers"]
        ]
        names = [ld["name"] for ld in checkpoint["layers"]]

        model = cls(layers, names, name=checkpoint["name"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

    # ── diagnostics ───────────────────────────────────────────────────────────

    def grad_norm(self) -> float:
        """
        Global L2 gradient norm across all parameters.

        Call after loss.backward() and before the optimiser step to monitor
        gradient health. Values > 10 usually mean exploding gradients.
        """
        return math.sqrt(sum(p.grad ** 2 for p in self.parameters()))

    def weight_stats(self) -> dict[str, dict[str, float]]:
        """
        Per-layer descriptive statistics for weight magnitudes.

        Useful for spotting vanishing init (all near zero) or
        exploding init (very large values) before training begins.

        Returns:
            {
              'layer_0': {'mean': 0.012, 'std': 0.231, 'min': -0.45, 'max': 0.43},
              ...
            }
        """
        stats: dict[str, dict[str, float]] = {}
        for layer_name, layer in self._layers.items():
            weights = [w.data for n in layer.neurons for w in n.w]
            if not weights:
                continue
            n   = len(weights)
            mu  = sum(weights) / n
            std = math.sqrt(sum((w - mu) ** 2 for w in weights) / n)
            stats[layer_name] = {
                "mean": round(mu,  6),
                "std":  round(std, 6),
                "min":  round(min(weights), 6),
                "max":  round(max(weights), 6),
                "n":    n,
            }
        return stats

    # ── convenience ───────────────────────────────────────────────────────────

    def predict(self, x: Sequence[float]) -> float | list[float]:
        """
        Run a single sample in eval mode and return plain Python floats.

        Temporarily switches to eval mode so behaviour is consistent with
        inference, then restores the previous mode.

        Args:
            x: Input sample as plain floats.

        Returns:
            Single float for 1-output networks, list of floats otherwise.
        """
        was_training = self._training
        self.eval()

        out = self.forward(x)

        if was_training:
            self.train()

        if isinstance(out, Value):
            return out.data
        return [v.data for v in out]

    def __repr__(self) -> str:
        shapes = " → ".join(
            f"{l.n_inputs}[{l._activation_name}]" for l in self.layers
        ) + f" → {self.layers[-1].n_outputs}"
        return (
            f"Network(name={self.name!r}, arch={shapes}, "
            f"params={self.n_params}, mode={'train' if self._training else 'eval'})"
        )