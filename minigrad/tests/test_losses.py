import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
from engine.value import Value
from losses.losses import mse, bce, gradient_check

def approx_eq(a, b, tol=1e-6):
    return abs(a - b) < tol

# ── MSE Tests ─────────────────────────────────────────────────────

def test_mse_zero_loss():
    """Perfect prediction -> loss=0, grad=0"""
    p = Value(2.0)
    loss = mse([p], [2.0])
    assert approx_eq(loss.data, 0.0), f"expected 0, got {loss.data}"
    loss.backward()
    assert approx_eq(p.grad, 0.0), f"expected grad=0, got {p.grad}"
    print("MSE T1 passed — zero loss, zero grad")

def test_mse_known_value():
    """mse([3.0], [1.0]) = 4.0, grad = 4.0"""
    p = Value(3.0)
    loss = mse([p], [1.0])
    assert approx_eq(loss.data, 4.0), f"expected 4.0, got {loss.data}"
    loss.backward()
    # ∂L/∂p = 2/n * (p - t) = 2/1 * (3-1) = 4.0
    assert approx_eq(p.grad, 4.0), f"expected grad=4.0, got {p.grad}"
    print("MSE T2 passed — known value and gradient")

def test_mse_two_samples():
    """mse([0.0, 1.0], [1.0, 1.0]) = 0.5"""
    p0, p1 = Value(0.0), Value(1.0)
    loss = mse([p0, p1], [1.0, 1.0])
    assert approx_eq(loss.data, 0.5), f"expected 0.5, got {loss.data}"
    loss.backward()
    # grad p0 = 2/2*(0-1) = -1.0
    # grad p1 = 2/2*(1-1) = 0.0
    assert approx_eq(p0.grad, -1.0), f"p0.grad: {p0.grad}"
    assert approx_eq(p1.grad,  0.0), f"p1.grad: {p1.grad}"
    print("MSE T3 passed — two samples")

def test_mse_gradient_sign():
    """
    Catches pred - target vs target - pred sign error.
    pred < target -> gradient must be negative (push pred up).
    pred > target -> gradient must be positive (push pred down).
    """
    p_low  = Value(0.1)
    p_high = Value(0.9)
    loss_low  = mse([p_low],  [1.0])
    loss_high = mse([p_high], [0.0])
    loss_low.backward()
    loss_high.backward()
    assert p_low.grad  < 0, f"pred < target -> grad should be negative: {p_low.grad}"
    assert p_high.grad > 0, f"pred > target -> grad should be positive: {p_high.grad}"
    print("MSE T4 passed — gradient sign correct")

def test_mse_gradient_check():
    preds = [Value(0.3), Value(0.7), Value(-0.1)]
    targets = [0.5, 0.5, 0.5]
    results = gradient_check(mse, preds, targets)
    for i, (a, n, err) in enumerate(results):
        print(f"  pred[{i}]: analytical={a:.6f}, numerical={n:.6f}, err={err:.2e}")
    print("MSE T5 passed — gradient check")

# ── BCE Tests ─────────────────────────────────────────────────────

def test_bce_known_value():
    """bce([0.5], [1.0]) = -log(0.5) = log(2) ≈ 0.6931"""
    p = Value(0.5)
    loss = bce([p], [1.0])
    assert approx_eq(loss.data, math.log(2), tol=1e-5), \
        f"expected {math.log(2):.4f}, got {loss.data:.4f}"
    print("BCE T1 passed — known value")

def test_bce_near_zero_prediction():
    """bce([~0], [1.0]) should be large positive, not crash"""
    p = Value(1e-8)   # effectively 0
    loss = bce([p], [1.0])
    assert loss.data > 10.0, f"expected large loss, got {loss.data}"
    loss.backward()   # must not raise
    print(f"BCE T2 passed — near-zero prediction, loss={loss.data:.2f}")

def test_bce_near_one_prediction():
    """bce([~1], [0.0]) should be large positive, not crash"""
    p = Value(1 - 1e-8)
    loss = bce([p], [0.0])
    assert loss.data > 10.0, f"expected large loss, got {loss.data}"
    loss.backward()
    print(f"BCE T3 passed — near-one prediction, loss={loss.data:.2f}")

def test_bce_gradient_sign():
    """
    y=1, pred=0.3: gradient should be negative (push pred up toward 1)
    y=0, pred=0.7: gradient should be positive (push pred down toward 0)
    """
    p1 = Value(0.3)
    loss1 = bce([p1], [1.0])
    loss1.backward()
    assert p1.grad < 0, f"y=1, pred=0.3 -> grad should be negative: {p1.grad}"

    p2 = Value(0.7)
    loss2 = bce([p2], [0.0])
    loss2.backward()
    assert p2.grad > 0, f"y=0, pred=0.7 -> grad should be positive: {p2.grad}"
    print("BCE T4 passed — gradient sign correct")

def test_bce_gradient_check():
    preds   = [Value(0.3), Value(0.7), Value(0.5)]
    targets = [1.0, 0.0, 1.0]
    results = gradient_check(bce, preds, targets)
    for i, (a, n, err) in enumerate(results):
        print(f"  pred[{i}]: analytical={a:.6f}, numerical={n:.6f}, err={err:.2e}")
    print("BCE T5 passed — gradient check")

def test_loss_is_scalar():
    """Loss must be a single Value, not a list"""
    preds   = [Value(0.5), Value(0.3)]
    targets = [1.0, 0.0]
    mse_loss = mse(preds, targets)
    bce_loss = bce(preds, targets)
    assert isinstance(mse_loss, Value), "MSE must return a Value"
    assert isinstance(bce_loss, Value), "BCE must return a Value"
    # Must have a .backward() — single root
    mse_loss.backward()
    bce_loss.backward()
    print("T6 passed — loss is scalar Value with working backward")


if __name__ == '__main__':
    test_mse_zero_loss()
    test_mse_known_value()
    test_mse_two_samples()
    test_mse_gradient_sign()
    test_mse_gradient_check()
    test_bce_known_value()
    test_bce_near_zero_prediction()
    test_bce_near_one_prediction()
    test_bce_gradient_sign()
    test_bce_gradient_check()
    test_loss_is_scalar()
    print("\nAll loss tests passed.")