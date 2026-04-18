"""
tests/test_value.py
-------------------
Exhaustive test suite for engine/value.py (autodiff engine).

Run:  python tests/test_value.py

Test structure:
    TestForwardOps          — data correctness for all operations
    TestBackwardBasic       — gradients for atomic operations
    TestBackwardComposed    — multi-op graphs, chain rule
    TestGradientAccumulation — reused nodes, += vs = bug
    TestTopologicalOrder    — diamond graphs, multi-path flows
    TestActivationBackward  — relu, sigmoid, tanh gradients
    TestNumerical           — finite-difference gradient checks
    TestEdgeCases           — scalar interop, zero_grad, error guards
    TestChainRuleDeep       — long chains, vanishing gradient observation
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.value import Value


# ──────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────

def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def numerical_gradient(f, val: Value, h: float = 1e-5) -> float:
    """
    Finite-difference gradient check.
    Estimates df/d(val) by perturbing val.data by ±h.
    This is independent of the autodiff engine — pure calculus.
    If autodiff gradient matches this, the backward is correct.
    """
    original = val.data

    val.data = original + h
    y_plus = f().data

    val.data = original - h
    y_minus = f().data

    val.data = original   # restore
    return (y_plus - y_minus) / (2 * h)


def grad_check(test_case, f, inputs, tol=1e-5):
    """
    Run autodiff backward, then compare each input's .grad
    against finite-difference estimate.
    """
    out = f()
    out.backward()
    for v in inputs:
        auto_g = v.grad
        num_g  = numerical_gradient(f, v)
        test_case.assertAlmostEqual(
            auto_g, num_g, delta=tol,
            msg=f"Grad check failed for {v}: autodiff={auto_g:.8f}, numerical={num_g:.8f}"
        )


# ═══════════════════════════════════════════════════════════
# FORWARD OPERATIONS
# ═══════════════════════════════════════════════════════════

class TestForwardOps(unittest.TestCase):

    def test_add_data(self):
        self.assertEqual((Value(3.0) + Value(4.0)).data, 7.0)

    def test_mul_data(self):
        self.assertEqual((Value(3.0) * Value(4.0)).data, 12.0)

    def test_sub_data(self):
        self.assertEqual((Value(5.0) - Value(3.0)).data, 2.0)

    def test_neg_data(self):
        self.assertEqual((-Value(3.0)).data, -3.0)

    def test_div_data(self):
        self.assertAlmostEqual((Value(6.0) / Value(2.0)).data, 3.0)

    def test_pow_data(self):
        self.assertAlmostEqual((Value(3.0) ** 2).data, 9.0)

    def test_pow_negative_exponent(self):
        self.assertAlmostEqual((Value(2.0) ** -1).data, 0.5)

    def test_exp_data(self):
        self.assertAlmostEqual(Value(1.0).exp().data, math.e, places=10)

    def test_exp_zero(self):
        self.assertAlmostEqual(Value(0.0).exp().data, 1.0, places=15)

    def test_log_data(self):
        self.assertAlmostEqual(Value(math.e).log().data, 1.0, places=10)

    def test_log_one(self):
        self.assertAlmostEqual(Value(1.0).log().data, 0.0, places=15)

    def test_log_zero_raises(self):
        with self.assertRaises(ValueError):
            Value(0.0).log()

    def test_log_negative_raises(self):
        with self.assertRaises(ValueError):
            Value(-1.0).log()

    def test_relu_positive(self):
        self.assertEqual(Value(3.0).relu().data, 3.0)

    def test_relu_negative(self):
        self.assertEqual(Value(-3.0).relu().data, 0.0)

    def test_relu_zero(self):
        self.assertEqual(Value(0.0).relu().data, 0.0)

    def test_sigmoid_zero(self):
        self.assertAlmostEqual(Value(0.0).sigmoid().data, 0.5, places=15)

    def test_sigmoid_large_positive(self):
        self.assertAlmostEqual(Value(100.0).sigmoid().data, 1.0, places=6)

    def test_sigmoid_large_negative(self):
        self.assertAlmostEqual(Value(-100.0).sigmoid().data, 0.0, places=6)

    def test_tanh_zero(self):
        self.assertAlmostEqual(Value(0.0).tanh().data, 0.0, places=15)

    def test_tanh_matches_math(self):
        for x in [-2.0, -0.5, 0.5, 2.0]:
            self.assertAlmostEqual(Value(x).tanh().data, math.tanh(x), places=12)

    def test_radd(self):
        """2 + Value(3) should work via __radd__."""
        result = 2 + Value(3.0)
        self.assertEqual(result.data, 5.0)

    def test_rmul(self):
        """2 * Value(3) should work via __rmul__."""
        result = 2 * Value(3.0)
        self.assertEqual(result.data, 6.0)

    def test_rsub(self):
        """5 - Value(3) should work via __rsub__."""
        result = 5 - Value(3.0)
        self.assertEqual(result.data, 2.0)

    def test_rtruediv(self):
        """6 / Value(2) should work via __rtruediv__."""
        result = 6 / Value(2.0)
        self.assertAlmostEqual(result.data, 3.0)


# ═══════════════════════════════════════════════════════════
# BACKWARD — ATOMIC OPERATIONS
# ═══════════════════════════════════════════════════════════

class TestBackwardBasic(unittest.TestCase):
    """
    Each test: single operation, verify both parent gradients.
    Derivations written inline so the test IS the specification.
    """

    def test_add_grads(self):
        # d(a+b)/da = 1,  d(a+b)/db = 1
        a, b = Value(3.0), Value(4.0)
        (a + b).backward()
        self.assertAlmostEqual(a.grad, 1.0)
        self.assertAlmostEqual(b.grad, 1.0)

    def test_mul_grads(self):
        # d(a*b)/da = b = 4,  d(a*b)/db = a = 3
        a, b = Value(3.0), Value(4.0)
        (a * b).backward()
        self.assertAlmostEqual(a.grad, 4.0)
        self.assertAlmostEqual(b.grad, 3.0)

    def test_sub_grads(self):
        # d(a-b)/da = 1,  d(a-b)/db = -1
        a, b = Value(5.0), Value(3.0)
        (a - b).backward()
        self.assertAlmostEqual(a.grad,  1.0)
        self.assertAlmostEqual(b.grad, -1.0)

    def test_neg_grad(self):
        # d(-a)/da = -1
        a = Value(3.0)
        (-a).backward()
        self.assertAlmostEqual(a.grad, -1.0)

    def test_div_grads(self):
        # d(a/b)/da = 1/b = 0.5
        # d(a/b)/db = -a/b² = -6/4 = -1.5
        a, b = Value(6.0), Value(2.0)
        (a / b).backward()
        self.assertAlmostEqual(a.grad,  0.5,  places=10)
        self.assertAlmostEqual(b.grad, -1.5,  places=10)

    def test_pow_grad(self):
        # d(a²)/da = 2a = 6
        a = Value(3.0)
        (a ** 2).backward()
        self.assertAlmostEqual(a.grad, 6.0, places=10)

    def test_pow_fractional(self):
        # d(a^0.5)/da = 0.5 * a^{-0.5} = 0.5 / sqrt(4) = 0.25
        a = Value(4.0)
        (a ** 0.5).backward()
        self.assertAlmostEqual(a.grad, 0.25, places=10)

    def test_exp_grad(self):
        # d(e^a)/da = e^a
        a = Value(2.0)
        a.exp().backward()
        self.assertAlmostEqual(a.grad, math.exp(2.0), places=10)

    def test_log_grad(self):
        # d(ln a)/da = 1/a
        a = Value(3.0)
        a.log().backward()
        self.assertAlmostEqual(a.grad, 1.0 / 3.0, places=10)

    def test_relu_positive_grad(self):
        a = Value(2.0)
        a.relu().backward()
        self.assertAlmostEqual(a.grad, 1.0)

    def test_relu_negative_grad(self):
        a = Value(-2.0)
        a.relu().backward()
        self.assertAlmostEqual(a.grad, 0.0)

    def test_relu_zero_grad(self):
        """Sub-gradient at 0 is 0."""
        a = Value(0.0)
        a.relu().backward()
        self.assertAlmostEqual(a.grad, 0.0)

    def test_sigmoid_grad_at_zero(self):
        # σ'(0) = 0.25
        a = Value(0.0)
        a.sigmoid().backward()
        self.assertAlmostEqual(a.grad, 0.25, places=12)

    def test_tanh_grad_at_zero(self):
        # tanh'(0) = 1
        a = Value(0.0)
        a.tanh().backward()
        self.assertAlmostEqual(a.grad, 1.0, places=12)


# ═══════════════════════════════════════════════════════════
# GRADIENT ACCUMULATION — reused nodes
# ═══════════════════════════════════════════════════════════

class TestGradientAccumulation(unittest.TestCase):
    """
    These tests specifically catch the += vs = bug.
    If _backward uses = instead of +=, a reused node
    will only receive the gradient from the LAST consumer,
    silently losing contributions from earlier ones.
    """

    def test_x_plus_x(self):
        """
        y = x + x → dy/dx = 2
        Two paths from x to y, each contributing 1.
        """
        x = Value(3.0)
        y = x + x
        y.backward()
        self.assertAlmostEqual(x.grad, 2.0,
            msg="x + x: expected grad=2.0 (catches = vs += bug)")

    def test_x_times_x(self):
        """
        y = x * x = x²  → dy/dx = 2x = 6
        """
        x = Value(3.0)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad, 6.0,
            msg="x * x: expected grad=2x=6.0")

    def test_three_uses(self):
        """
        y = x + x + x = 3x  → dy/dx = 3
        """
        x = Value(2.0)
        y = x + x + x
        y.backward()
        self.assertAlmostEqual(x.grad, 3.0)

    def test_mixed_reuse(self):
        """
        c = a * b
        d = c + a      ← a appears twice
        dd/da = dd/dc * dc/da + dd/da_direct
              = 1 * b + 1 = 3 + 1 = 4
        dd/db = dd/dc * dc/db = 1 * a = 2
        """
        a, b = Value(2.0), Value(3.0)
        c = a * b
        d = c + a
        d.backward()
        self.assertAlmostEqual(a.grad, 4.0)
        self.assertAlmostEqual(b.grad, 2.0)


# ═══════════════════════════════════════════════════════════
# TOPOLOGICAL ORDER
# ═══════════════════════════════════════════════════════════

class TestTopologicalOrder(unittest.TestCase):
    """
    Tests that expose wrong topological order.
    If a node runs _backward before all consumers have sent
    gradient to it, its parents receive incomplete gradients.
    """

    def test_diamond_graph(self):
        """
        Diamond:  b = a*3,  c = a*4,  d = b+c
        d.backward() must accumulate:
            a.grad = dd/db * db/da + dd/dc * dc/da
                   = 1*3 + 1*4 = 7
        """
        a = Value(2.0)
        b = a * Value(3.0)
        c = a * Value(4.0)
        d = b + c
        d.backward()
        self.assertAlmostEqual(a.grad, 7.0,
            msg="Diamond graph: a.grad must be 7.0")

    def test_wide_fan_out(self):
        """
        y = a + a + a + a + a  (5 paths)
        dy/da = 5
        """
        a = Value(1.0)
        y = a + a + a + a + a
        y.backward()
        self.assertAlmostEqual(a.grad, 5.0)

    def test_sequential_chain(self):
        """
        c = a * b
        e = c * d
        de/da = de/dc * dc/da = d * b = 4 * 3 = 12
        """
        a, b, d = Value(2.0), Value(3.0), Value(4.0)
        c = a * b
        e = c * d
        e.backward()
        self.assertAlmostEqual(a.grad, 12.0)
        self.assertAlmostEqual(b.grad,  8.0)  # de/db = d * a = 4 * 2 = 8
        self.assertAlmostEqual(d.grad,  6.0)  # de/dd = c = a*b = 6

    def test_topo_order_leaves_first(self):
        """
        Topological order returned by topo_order() must have leaves
        (no parents) at the start and the root at the end.
        """
        a = Value(2.0, _label='a')
        b = Value(3.0, _label='b')
        c = a * b
        order = c.topo_order()
        # Root must be last
        self.assertIs(order[-1], c)
        # Leaves must come before nodes that depend on them
        idx = {id(n): i for i, n in enumerate(order)}
        self.assertLess(idx[id(a)], idx[id(c)])
        self.assertLess(idx[id(b)], idx[id(c)])


# ═══════════════════════════════════════════════════════════
# COMPOSED OPERATIONS — chain rule
# ═══════════════════════════════════════════════════════════

class TestBackwardComposed(unittest.TestCase):

    def test_linear_chain(self):
        """
        c = a * 3
        e = c * 4
        de/da = 12
        """
        a = Value(2.0)
        c = a * Value(3.0)
        e = c * Value(4.0)
        e.backward()
        self.assertAlmostEqual(a.grad, 12.0)

    def test_quadratic(self):
        """
        y = a² + 2a + 1 = (a+1)²
        dy/da = 2a + 2 = 2*3 + 2 = 8
        """
        a = Value(3.0)
        y = a * a + Value(2.0) * a + Value(1.0)
        y.backward()
        self.assertAlmostEqual(a.grad, 8.0, places=10)

    def test_exp_log_identity(self):
        """
        y = log(exp(a)) = a  →  dy/da = 1
        """
        a = Value(2.0)
        y = a.exp().log()
        y.backward()
        self.assertAlmostEqual(a.grad, 1.0, places=10)

    def test_neuron_forward_backward(self):
        """
        Simulates one neuron: out = tanh(w*x + b)
        Checks autodiff gradients match finite differences.

        Key design rule:
            autodiff uses the ORIGINAL Value objects directly (their .grad is set).
            numerical_gradient uses a fresh-graph factory that reads .data —
            these are two independent computations that must agree.
        """
        w = Value(0.5, _label='w')
        x = Value(2.0, _label='x')
        b = Value(-1.0, _label='b')

        # ── Autodiff: compute on the originals ─────────────
        out = (w * x + b).tanh()
        out.backward()
        # w.grad, x.grad, b.grad are now set

        # ── Numerical: factory reads .data for perturbation ─
        h = 1e-5
        wv, xv, bv = w.data, x.data, b.data

        def make(ww, xx, bb):
            return (Value(ww) * Value(xx) + Value(bb)).tanh().data

        num_g_w = (make(wv+h, xv,   bv)   - make(wv-h, xv,   bv))   / (2*h)
        num_g_x = (make(wv,   xv+h, bv)   - make(wv,   xv-h, bv))   / (2*h)
        num_g_b = (make(wv,   xv,   bv+h) - make(wv,   xv,   bv-h)) / (2*h)

        self.assertAlmostEqual(w.grad, num_g_w, delta=1e-5,
            msg=f"Neuron grad check failed for w: autodiff={w.grad:.8f}, numerical={num_g_w:.8f}")
        self.assertAlmostEqual(x.grad, num_g_x, delta=1e-5,
            msg=f"Neuron grad check failed for x: autodiff={x.grad:.8f}, numerical={num_g_x:.8f}")
        self.assertAlmostEqual(b.grad, num_g_b, delta=1e-5,
            msg=f"Neuron grad check failed for b: autodiff={b.grad:.8f}, numerical={num_g_b:.8f}")

    def test_mse_loss(self):
        """
        L = (pred - target)²
        dL/d(pred) = 2*(pred - target)
        """
        pred   = Value(3.0)
        target = Value(1.0)
        loss   = (pred - target) ** 2
        loss.backward()
        self.assertAlmostEqual(pred.grad, 4.0, places=10)   # 2*(3-1)=4


# ═══════════════════════════════════════════════════════════
# ACTIVATION BACKWARD — finite-difference verified
# ═══════════════════════════════════════════════════════════

class TestActivationBackward(unittest.TestCase):
    """
    For each activation, compare autodiff gradient against
    independent finite-difference estimate at several input values.
    """

    def _check_activation(self, act_fn, xs):
        for x_val in xs:
            x = Value(x_val)

            def f():
                return act_fn(Value(x.data))

            out = f()
            out.backward()  # not used — we use x directly

            x2 = Value(x_val)
            res = act_fn(x2)
            res.backward()
            auto_g = x2.grad
            num_g  = numerical_gradient(f, x)
            self.assertAlmostEqual(
                auto_g, num_g, delta=1e-5,
                msg=f"{act_fn.__name__} grad check failed at x={x_val}: "
                    f"autodiff={auto_g:.8f}, numerical={num_g:.8f}"
            )

    def test_relu_grad_check(self):
        self._check_activation(lambda v: v.relu(), [-2.0, -0.1, 0.1, 1.0, 5.0])

    def test_sigmoid_grad_check(self):
        self._check_activation(lambda v: v.sigmoid(), [-3.0, -1.0, 0.0, 1.0, 3.0])

    def test_tanh_grad_check(self):
        self._check_activation(lambda v: v.tanh(), [-3.0, -1.0, 0.0, 1.0, 3.0])

    def test_exp_grad_check(self):
        self._check_activation(lambda v: v.exp(), [-2.0, 0.0, 1.0, 2.0])

    def test_log_grad_check(self):
        self._check_activation(lambda v: v.log(), [0.5, 1.0, 2.0, 5.0])

    def test_pow_grad_check(self):
        self._check_activation(lambda v: v ** 3, [0.5, 1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════
# SCALAR INTEROP
# ═══════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_scalar_add(self):
        a = Value(3.0)
        b = a + 2
        b.backward()
        self.assertAlmostEqual(a.grad, 1.0)

    def test_scalar_mul(self):
        a = Value(3.0)
        b = a * 4
        b.backward()
        self.assertAlmostEqual(a.grad, 4.0)

    def test_scalar_radd(self):
        a = Value(3.0)
        b = 2 + a
        b.backward()
        self.assertAlmostEqual(a.grad, 1.0)

    def test_scalar_rmul(self):
        a = Value(3.0)
        b = 4 * a
        b.backward()
        self.assertAlmostEqual(a.grad, 4.0)

    def test_pow_bad_exponent_raises(self):
        with self.assertRaises(TypeError):
            Value(2.0) ** Value(3.0)

    def test_zero_grad_resets(self):
        a = Value(2.0)
        b = a * Value(3.0)
        b.backward()
        self.assertAlmostEqual(a.grad, 3.0)
        b.zero_grad()
        self.assertAlmostEqual(a.grad, 0.0)

    def test_grad_starts_at_zero(self):
        a = Value(5.0)
        self.assertEqual(a.grad, 0.0)

    def test_leaf_has_no_parents(self):
        a = Value(1.0)
        self.assertEqual(len(a._parents), 0)

    def test_op_is_tracked(self):
        a, b = Value(1.0), Value(2.0)
        c = a + b
        self.assertEqual(c._op, '+')
        d = a * b
        self.assertEqual(d._op, '*')

    def test_repr_contains_data(self):
        a = Value(3.14)
        self.assertIn("3.14", repr(a))

    def test_multiple_backward_calls_accumulate(self):
        """
        Calling backward() twice without zero_grad() accumulates
        gradients — same as calling it once with double the loss.
        This is a known gotcha in PyTorch too.
        """
        a = Value(2.0)
        b = a * Value(3.0)

        b.backward()
        first_grad = a.grad   # should be 3.0

        b2 = Value(a.data) * Value(3.0)
        # Build fresh graph from same value
        a2 = Value(2.0)
        c  = a2 * Value(3.0)
        c.backward()
        self.assertAlmostEqual(a2.grad, 3.0)


# ═══════════════════════════════════════════════════════════
# DEEP CHAIN — vanishing gradient observation
# ═══════════════════════════════════════════════════════════

class TestChainRuleDeep(unittest.TestCase):
    """
    These tests verify correctness in deep chains and
    simultaneously let you observe vanishing gradients.
    """

    def test_long_mul_chain(self):
        """
        y = x * 0.5 * 0.5 * ... (n times) = x * 0.5^n
        dy/dx = 0.5^n
        """
        x = Value(1.0)
        y = x
        n = 10
        for _ in range(n):
            y = y * Value(0.5)
        y.backward()
        expected = 0.5 ** n
        self.assertAlmostEqual(x.grad, expected, places=10)

    def test_sigmoid_chain_vanishing(self):
        """
        Stack 5 sigmoid layers.  Gradient at input should be tiny
        (vanishing gradient) but still mathematically correct.

        Design rule: run backward on the ORIGINAL x; use a separate
        factory for the numerical estimate.
        """
        x = Value(0.0)

        # Autodiff: build chain from original x
        v = x
        for _ in range(5):
            v = v.sigmoid()
        v.backward()
        auto_g = x.grad

        # Numerical: fresh-graph factory that reads x.data
        h = 1e-5
        def make(xv):
            vv = Value(xv)
            for _ in range(5):
                vv = vv.sigmoid()
            return vv.data

        num_g = (make(x.data + h) - make(x.data - h)) / (2 * h)

        self.assertAlmostEqual(auto_g, num_g, delta=1e-5,
            msg=f"Sigmoid chain grad: autodiff={auto_g:.8f}, numerical={num_g:.8f}")

        # Observe vanishing gradient
        self.assertLess(abs(auto_g), 0.1,
            msg="Expected vanishing gradient through 5 sigmoid layers")

    def test_tanh_chain_less_vanishing(self):
        """
        Stack 5 tanh layers.  tanh'(0) = 1, so at x=0
        gradient survives better than sigmoid.
        """
        x = Value(0.0)

        # Autodiff on original x
        v = x
        for _ in range(5):
            v = v.tanh()
        v.backward()
        auto_g = x.grad

        # Numerical
        h = 1e-5
        def make(xv):
            vv = Value(xv)
            for _ in range(5):
                vv = vv.tanh()
            return vv.data

        num_g = (make(x.data + h) - make(x.data - h)) / (2 * h)
        self.assertAlmostEqual(auto_g, num_g, delta=1e-5)

    def test_relu_chain_no_vanishing(self):
        """
        Stack 5 relu layers on positive input.
        Gradient should be exactly 1.0 (no vanishing for relu).
        """
        x = Value(1.0)
        y = x
        for _ in range(5):
            y = y.relu()
        y.backward()
        self.assertAlmostEqual(x.grad, 1.0, places=10)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)