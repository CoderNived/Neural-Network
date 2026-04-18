import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import random
from engine.value import Value
from nn.neuron import Neuron

def approx_eq(a, b, tol=1e-4):
    return abs(a - b) < tol

# ── Test 1 — output is a Value in correct range ───────────────────
def test_output_type_and_range():
    n = Neuron(3, activation='tanh')
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value), "output must be a Value"
    assert -1.0 < out.data < 1.0,  f"tanh output out of range: {out.data}"
    print("T1 passed — output type and range")

# ── Test 2 — backward populates gradients ─────────────────────────
def test_backward_populates_grads():
    # Use fixed weights to guarantee non-saturation
    n = Neuron(3, activation='tanh')
    for w in n.w:
        w.data = 0.1   # small weights -> z small -> not saturated
    out = n([1.0, 2.0, 3.0])
    out.backward()
    for i, p in enumerate(n.parameters()):
        assert p.grad != 0.0, f"parameter {i} has zero grad"
    print("T2 passed — backward populates gradients")

# ── Test 3 — parameter count ──────────────────────────────────────
def test_parameter_count():
    n = Neuron(3, activation='tanh')
    assert len(n.parameters()) == 4, f"expected 4, got {len(n.parameters())}"
    n2 = Neuron(10, activation='relu')
    assert len(n2.parameters()) == 11
    print("T3 passed — parameter count")

# ── Test 4 — finite difference gradient check ─────────────────────
def test_finite_difference():
    """
    For each parameter p:
      numerical gradient = (f(p+ε) - f(p-ε)) / 2ε
    Compare to p.grad from backward().
    This is the ground truth test — catches wrong backward formulas.
    """
    random.seed(42)
    n = Neuron(3, activation='tanh')
    x = [0.5, -0.3, 0.8]
    eps = 1e-4

    # Analytical gradient via backward
    n.zero_grad()
    out = n(x)
    out.backward()
    analytical = [p.grad for p in n.parameters()]

    # Numerical gradient via finite differences
    params = n.parameters()
    numerical = []
    for p in params:
        orig = p.data

        p.data = orig + eps
        out_plus = n(x).data

        p.data = orig - eps
        out_minus = n(x).data

        p.data = orig  # restore
        numerical.append((out_plus - out_minus) / (2 * eps))

    for i, (a, num) in enumerate(zip(analytical, numerical)):
        assert approx_eq(a, num, tol=1e-3), \
            f"Gradient mismatch at param {i}: analytical={a:.6f}, numerical={num:.6f}"

    print("T4 passed — finite difference gradient check")

# ── Test 5 — accepts Value inputs ─────────────────────────────────
def test_value_inputs():
    """
    When neurons are stacked, outputs of one layer are Value objects
    fed as inputs to the next. The graph must stay connected.
    If we re-wrapped them as new Value objects, the graph would break
    and gradients wouldn't flow back through previous layers.
    """
    n = Neuron(2, activation='tanh')
    # plain float inputs
    out1 = n([1.0, 2.0])
    assert isinstance(out1, Value)

    # Value inputs — graph stays connected
    x1, x2 = Value(1.0), Value(2.0)
    n.zero_grad()
    out2 = n([x1, x2])
    out2.backward()
    # x1 and x2 should have gradients — they're part of the graph
    assert x1.grad != 0.0, "gradient didn't flow into Value input"
    assert x2.grad != 0.0, "gradient didn't flow into Value input"
    print("T5 passed — Value inputs, gradient flows through")

# ── Failure 1 — symmetric initialization experiment ───────────────
def test_symmetric_init():
    """
    All weights equal -> all gradients equal -> weights update identically.
    Network width is effectively 1 regardless of neuron count.
    """
    n = Neuron(4, activation='tanh')
    for w in n.w:
        w.data = 1.0   # symmetric

    out = n([1.0, 1.0, 1.0, 1.0])
    out.backward()

    grads = [w.grad for w in n.w]
    # All gradients are equal — this is the failure mode
    assert all(approx_eq(g, grads[0]) for g in grads), \
        "Expected equal grads under symmetric init"
    print(f"Failure 1 observed — all weight grads equal: {[f'{g:.4f}' for g in grads]}")
    print("  (This is the symmetry breaking failure — never init weights equally)")

# ── Failure 2 — parameters() returns originals, not copies ────────
def test_parameters_are_originals():
    n = Neuron(3, activation='tanh')
    params = n.parameters()
    params[0].grad = 999.0
    assert n.w[0].grad == 999.0, \
        "parameters() returned a copy — optimizer would update the wrong objects"
    params[0].grad = 0.0  # restore
    print("T6 passed — parameters() returns original Value objects")

# ── Failure 3 — stale gradient detection ──────────────────────────
def test_stale_gradients():
    n = Neuron(3, activation='tanh')
    x = [1.0, 2.0, 3.0]

    # First pass
    out = n(x)
    out.backward()
    grads_first = [p.grad for p in n.parameters()]

    # Second pass WITHOUT zero_grad
    out = n(x)
    out.backward()
    grads_doubled = [p.grad for p in n.parameters()]

    # Gradients should be doubled (accumulated, not reset)
    for g1, g2 in zip(grads_first, grads_doubled):
        assert approx_eq(g2, 2 * g1, tol=1e-6), \
            f"Expected doubled grad: {g1:.4f} -> {g2:.4f}"

    # Now with zero_grad
    n.zero_grad()
    out = n(x)
    out.backward()
    grads_clean = [p.grad for p in n.parameters()]

    for g1, gc in zip(grads_first, grads_clean):
        assert approx_eq(g1, gc, tol=1e-6), \
            f"After zero_grad, expected clean grad: {gc:.4f} != {g1:.4f}"

    print("T7 passed — stale gradient and zero_grad behavior confirmed")


if __name__ == '__main__':
    test_output_type_and_range()
    test_backward_populates_grads()
    test_parameter_count()
    test_finite_difference()
    test_value_inputs()
    test_symmetric_init()
    test_parameters_are_originals()
    test_stale_gradients()
    print("\nAll neuron tests passed.")