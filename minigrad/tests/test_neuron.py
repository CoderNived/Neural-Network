"""
tests/test_neuron.py — pytest suite for nn.neuron.Neuron

Run:  pytest tests/test_neuron.py -v
"""

from __future__ import annotations

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.value import Value
from engine.ops   import ACTIVATIONS
from nn.neuron    import Neuron


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

EPS = 1e-4          # finite-difference step
GRAD_TOL = 1e-3     # tolerance for gradient agreement


def numerical_grad(neuron: Neuron, x: list[float], param_idx: int) -> float:
    """Central-difference gradient for one parameter."""
    params = neuron.parameters()
    p = params[param_idx]
    orig = p.data

    p.data = orig + EPS
    f_plus = neuron(x).data

    p.data = orig - EPS
    f_minus = neuron(x).data

    p.data = orig          # restore
    return (f_plus - f_minus) / (2 * EPS)


def analytical_grads(neuron: Neuron, x: list[float]) -> list[float]:
    """Run one forward + backward pass, return all parameter gradients."""
    neuron.zero_grad()
    out = neuron(x)
    out.backward()
    return [p.grad for p in neuron.parameters()]


def approx(a: float, b: float, tol: float = GRAD_TOL) -> bool:
    return abs(a - b) < tol


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def seed():
    """Fix RNG so every test is deterministic."""
    random.seed(0)


@pytest.fixture
def small_tanh() -> Neuron:
    """fan_in=3, tanh, small weights so output is never saturated."""
    n = Neuron(3, activation="tanh")
    for w in n.w:
        w.data = 0.1
    return n


@pytest.fixture
def x3() -> list[float]:
    return [0.5, -0.3, 0.8]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            Neuron(3, activation="gelu")

    def test_zero_fan_in_raises(self):
        with pytest.raises(ValueError):
            Neuron(0)

    def test_bias_true_adds_param(self):
        n = Neuron(4, bias=True)
        assert len(n.parameters()) == 5       # 4 weights + 1 bias
        assert n.b is not None

    def test_bias_false_removes_param(self):
        n = Neuron(4, bias=False)
        assert len(n.parameters()) == 4       # weights only
        assert n.b is None

    @pytest.mark.parametrize("fan_in", [1, 5, 64, 256])
    def test_parameter_count_scales_with_fan_in(self, fan_in):
        n = Neuron(fan_in, bias=True)
        assert len(n.parameters()) == fan_in + 1

    @pytest.mark.parametrize("activation", list(ACTIVATIONS))
    def test_all_activations_construct(self, activation):
        n = Neuron(3, activation=activation)
        assert n._activation_name == activation

    def test_weights_are_value_objects(self):
        n = Neuron(4)
        assert all(isinstance(w, Value) for w in n.w)

    def test_bias_initialised_at_zero(self):
        n = Neuron(4, bias=True)
        assert n.b.data == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Forward pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestForward:

    def test_output_is_value(self, small_tanh, x3):
        out = small_tanh(x3)
        assert isinstance(out, Value)

    def test_tanh_output_bounded(self, small_tanh, x3):
        out = small_tanh(x3)
        assert -1.0 < out.data < 1.0

    def test_sigmoid_output_bounded(self, x3):
        n = Neuron(3, activation="sigmoid")
        out = n(x3)
        assert 0.0 < out.data < 1.0

    def test_relu_output_non_negative(self, x3):
        n = Neuron(3, activation="relu")
        out = n(x3)
        assert out.data >= 0.0

    def test_wrong_input_length_raises(self, small_tanh):
        with pytest.raises(ValueError, match="Expected"):
            small_tanh([1.0, 2.0])            # fan_in=3, giving 2

    def test_accepts_int_inputs(self, small_tanh):
        out = small_tanh([1, 2, 3])           # ints, not floats
        assert isinstance(out, Value)

    def test_accepts_value_inputs(self):
        """Value inputs must stay in the graph — not re-wrapped as new nodes."""
        n = Neuron(2, activation="tanh")
        x1, x2 = Value(1.0), Value(2.0)
        out = n([x1, x2])
        out.backward()
        assert x1.grad != 0.0, "gradient did not flow back through Value input"
        assert x2.grad != 0.0, "gradient did not flow back through Value input"

    def test_mixed_float_value_inputs(self):
        n = Neuron(3, activation="tanh")
        v = Value(0.5)
        out = n([v, 1.0, 2.0])
        out.backward()
        assert v.grad != 0.0

    def test_each_call_builds_fresh_graph(self, small_tanh, x3):
        """Two forward passes must not share graph nodes."""
        out1 = small_tanh(x3)
        out2 = small_tanh(x3)
        assert out1 is not out2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Backward pass — gradient correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackward:

    @pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu",
                                             "leaky_relu", "elu", "swish"])
    def test_backward_populates_all_grads(self, activation):
        random.seed(1)
        n = Neuron(3, activation=activation)
        # Keep weights small so ReLU neurons aren't dead
        for w in n.w:
            w.data = 0.1
        x = [0.5, -0.2, 0.8]
        out = n(x)
        out.backward()
        for i, p in enumerate(n.parameters()):
            assert p.grad != 0.0, f"{activation}: param {i} has zero grad"

    @pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu",
                                             "leaky_relu", "elu", "swish"])
    def test_finite_difference_all_activations(self, activation):
        """
        Central-difference check for every activation.
        This is the ground-truth test: catches any wrong backward formula.
        """
        random.seed(2)
        n = Neuron(3, activation=activation)
        for w in n.w:
            w.data = 0.1       # avoid saturation / dead relu
        x = [0.5, -0.3, 0.8]

        analytical = analytical_grads(n, x)
        for i in range(len(n.parameters())):
            num = numerical_grad(n, x, i)
            assert approx(analytical[i], num), (
                f"{activation} param[{i}]: "
                f"analytical={analytical[i]:.6f}, numerical={num:.6f}"
            )

    def test_gradient_flows_to_chained_neuron(self):
        """
        Stacking two neurons: grad must flow all the way back to layer-1 weights.
        """
        random.seed(3)
        n1 = Neuron(2, activation="tanh")
        n2 = Neuron(2, activation="tanh")
        for w in n1.w + n2.w:
            w.data = 0.1

        h   = [n1([0.5, -0.3])]          # single hidden neuron
        out = n2([h[0], Value(1.0)])
        out.backward()

        for i, w in enumerate(n1.w):
            assert w.grad != 0.0, f"n1.w[{i}] grad did not flow through chain"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Gradient accumulation and zero_grad
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradAccumulation:

    def test_grads_accumulate_without_zero_grad(self, small_tanh, x3):
        out = small_tanh(x3)
        out.backward()
        g1 = [p.grad for p in small_tanh.parameters()]

        out = small_tanh(x3)
        out.backward()                    # no zero_grad between passes
        g2 = [p.grad for p in small_tanh.parameters()]

        for a, b in zip(g1, g2):
            assert approx(b, 2 * a, tol=1e-6), \
                f"Expected doubled gradient: {a:.6f} -> {b:.6f}"

    def test_zero_grad_resets_all(self, small_tanh, x3):
        small_tanh(x3).backward()
        small_tanh.zero_grad()
        assert all(p.grad == 0.0 for p in small_tanh.parameters())

    def test_grad_clean_after_zero_grad(self, small_tanh, x3):
        small_tanh(x3).backward()
        g1 = [p.grad for p in small_tanh.parameters()]

        small_tanh.zero_grad()
        small_tanh(x3).backward()
        g2 = [p.grad for p in small_tanh.parameters()]

        for a, b in zip(g1, g2):
            assert approx(a, b, tol=1e-6), \
                f"After zero_grad pass, grad mismatch: {a:.6f} vs {b:.6f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Parameter identity
# ═══════════════════════════════════════════════════════════════════════════════

class TestParameterIdentity:

    def test_parameters_are_same_objects(self):
        """parameters() must return the original Value nodes, not copies."""
        n = Neuron(3)
        params = n.parameters()
        params[0].grad = 999.0
        assert n.w[0].grad == 999.0, "parameters() returned copies — optimizer would miss real weights"
        params[0].grad = 0.0

    def test_parameters_stable_across_calls(self, small_tanh, x3):
        """The same weight objects must appear in parameters() every call."""
        p1 = n.parameters() if (n := small_tanh) else None
        small_tanh(x3)
        p2 = small_tanh.parameters()
        assert all(a is b for a, b in zip(p1, p2))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Pathological / failure-mode tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureModes:

    def test_symmetry_breaking_failure(self):
        """
        Identical weights → identical gradients → zero effective width.
        This test documents the failure mode; it should always pass.
        """
        n = Neuron(4, activation="tanh")
        for w in n.w:
            w.data = 1.0           # symmetric initialisation

        n([1.0, 1.0, 1.0, 1.0]).backward()
        grads = [w.grad for w in n.w]

        assert all(approx(g, grads[0]) for g in grads), (
            "Symmetric init must produce equal gradients — "
            "documents why random init is required"
        )

    def test_relu_dead_neuron(self):
        """Negative pre-activation → ReLU output = 0 → zero gradient."""
        n = Neuron(2, activation="relu")
        for w in n.w:
            w.data = -1.0
        if n.b is not None:
            n.b.data = -10.0       # force strongly negative pre-activation

        out = n([1.0, 1.0])
        out.backward()

        assert out.data == 0.0,   "dead ReLU must output exactly 0"
        assert all(p.grad == 0.0 for p in n.parameters()), \
            "dead ReLU must produce zero gradients"

    def test_tanh_saturation(self):
        """Very large pre-activation → tanh ≈ ±1 → near-zero gradient."""
        n = Neuron(1, activation="tanh")
        n.w[0].data = 100.0
        if n.b is not None:
            n.b.data = 0.0

        out = n([1.0])
        out.backward()

        assert abs(out.data) > 0.999,      "should be saturated"
        assert abs(n.w[0].grad) < 1e-6,   "saturated tanh gradient should be ~0"

    def test_large_fan_in_output_stable(self):
        """He / Xavier init must prevent exploding pre-activations at large fan-in."""
        random.seed(7)
        n = Neuron(1024, activation="relu")
        x = [random.gauss(0, 1) for _ in range(1024)]
        out = n(x)
        assert math.isfinite(out.data), f"Output not finite: {out.data}"