"""
losses/losses.py
----------------
Loss functions for minigrad.

Every loss function:
    - Takes predictions (list of Value) and targets (list of float)
    - Returns a single scalar Value — the root of the computation graph
    - Is fully differentiable: calling .backward() on the result populates
      .grad on every prediction Value

Probabilistic interpretations (why these losses and not others):
    MSE    — maximum likelihood under Gaussian noise: argmin_θ ||y - f(x)||²
    BCE    — maximum likelihood under Bernoulli model: cross-entropy of
             predicted distribution P(y=1|x) against true label
    Hinge  — maximum-margin classification (SVM objective):
             penalises predictions that violate the margin, zero otherwise

Gradient summary (derived by hand, verified by gradient_check):
    MSE:   ∂L/∂pᵢ  =  (2/n) * (pᵢ - tᵢ)
    BCE:   ∂L/∂pᵢ  =  (1/n) * (-yᵢ/pᵢ + (1-yᵢ)/(1-pᵢ))
    Hinge: ∂L/∂pᵢ  =  -yᵢ/n  if yᵢ*pᵢ < 1  else  0
"""

import math
from engine.value import Value


# ═══════════════════════════════════════════════════════════
# INTERNAL GRAPH PRIMITIVES
# ═══════════════════════════════════════════════════════════

def _ensure_value(v) -> Value:
    return v if isinstance(v, Value) else Value(float(v))


def _safe_log(v: Value, eps: float = 1e-7) -> Value:
    """
    log(v) as a differentiable graph node with numerical clamping.

    Forward:  log(clamp(v.data, eps, 1-eps))
    Backward: 1 / clamp(v.data, eps, 1-eps)

    Why clamp in both directions (not just from below)?
        BCE uses both log(p) and log(1-p). If p is clamped to 1-eps
        for log(p), then 1-p = eps and log(1-p) is also safe. Clamping
        symmetrically ensures both terms are always finite.

    Why use clamped value in backward too?
        Consistency. If we used the raw value in backward but clamped
        in forward, the gradient would not correspond to the function
        we actually computed. Using the same clamped value keeps
        forward and backward mathematically coherent.
    """
    v = _ensure_value(v)
    c = max(eps, min(1.0 - eps, v.data))
    out = Value(math.log(c), _parents=(v,), _op='safe_log')

    def _backward():
        v.grad += (1.0 / c) * out.grad

    out._backward = _backward
    return out


def _safe_log_1minus(v: Value, eps: float = 1e-7) -> Value:
    """
    log(1 - v) as a differentiable graph node.

    Forward:  log(1 - clamp(v.data, eps, 1-eps))
    Backward: -1 / (1 - clamp(v.data, eps, 1-eps))
    """
    v = _ensure_value(v)
    c = max(eps, min(1.0 - eps, v.data))
    out = Value(math.log(1.0 - c), _parents=(v,), _op='safe_log1m')

    def _backward():
        v.grad += (-1.0 / (1.0 - c)) * out.grad

    out._backward = _backward
    return out


def _hinge_term(v: Value, y: float) -> Value:
    """
    max(0, 1 - y * pred) as a differentiable graph node.

    y must be in {-1, +1}.

    Gradient derivation:
        Let m = 1 - y*pred  (the margin violation)
        f(pred) = max(0, m)

        If m > 0  (margin violated):
            df/d(pred) = d/d(pred)[1 - y*pred] = -y
        If m < 0  (margin satisfied):
            df/d(pred) = 0
        If m == 0 (exactly on boundary):
            Subgradient convention: 0  (consistent with hinge = relu(-m))

    Why -y and not +y?
        Gradient flows upstream as: v.grad += (-y) * out.grad
        When y=+1 and pred is too small (margin violated), we need to
        INCREASE pred — the gradient w.r.t. pred must be negative so
        that gradient descent (subtract gradient) increases it. -y = -1
        gives a negative contribution, which is correct.
    """
    v = _ensure_value(v)
    margin = 1.0 - y * v.data
    out = Value(max(0.0, margin), _parents=(v,), _op='hinge')

    def _backward():
        if margin > 0:
            v.grad += (-y) * out.grad
        # margin <= 0: zero subgradient, nothing to accumulate

    out._backward = _backward
    return out


def _validate_lengths(predictions, targets, fn_name: str) -> None:
    if len(predictions) != len(targets):
        raise ValueError(
            f"{fn_name}: length mismatch — "
            f"{len(predictions)} predictions, {len(targets)} targets"
        )


# ═══════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════

def mse(predictions, targets) -> Value:
    """
    Mean Squared Error:  L = (1/n) * Σ (predᵢ - targetᵢ)²

    Gradient: ∂L/∂predᵢ = (2/n) * (predᵢ - targetᵢ)

    Geometric interpretation:
        The gradient is a vector from target toward prediction.
        Gradient descent subtracts it, pulling prediction toward target.
        The magnitude is proportional to the error — quadratic bowl,
        smooth and symmetric everywhere.

    Probabilistic basis:
        MSE = negative log-likelihood under Gaussian noise model.
        Minimising MSE assumes errors are Gaussian around the true value.

    Args:
        predictions: list of Value
        targets:     list of float or Value
    Returns:
        scalar Value
    """
    _validate_lengths(predictions, targets, 'mse')
    n       = len(predictions)
    targets = [_ensure_value(t) for t in targets]
    total   = Value(0.0)

    for p, t in zip(predictions, targets):
        diff  = p + (t * -1.0)      # pred - target, stays in graph
        total = total + diff * diff

    return total * (1.0 / n)


def bce(predictions, targets, eps: float = 1e-7) -> Value:
    """
    Binary Cross-Entropy:
        L = -(1/n) * Σ [yᵢ*log(pᵢ) + (1-yᵢ)*log(1-pᵢ)]

    Gradient: ∂L/∂pᵢ = (1/n) * (-yᵢ/pᵢ + (1-yᵢ)/(1-pᵢ))

    Why gradient is large when prediction is wrong:
        For y=1: gradient = -1/(n*p). As p→0 (maximally wrong), gradient→-∞.
        The loss surface becomes infinitely steep when confidently wrong.
        This is a feature: BCE is derived from Bernoulli MLE, and the log
        naturally produces this inverse relationship. MSE has no such urgency.

    Why sigmoid must precede BCE:
        BCE requires p ∈ (0,1). Raw logits are rejected with a clear error.
        The sigmoid+BCE combination is also mathematically elegant: the
        combined gradient simplifies to (pred - target), which is linear,
        never vanishing — BCE compensates exactly for sigmoid's saturation.

    Args:
        predictions: list of Value with .data in [0, 1] (post-sigmoid)
        targets:     list of float, binary labels in {0, 1}
        eps:         clamping value for log stability (default 1e-7)
    Returns:
        scalar Value
    """
    _validate_lengths(predictions, targets, 'bce')

    # Reject raw logits explicitly — better error than silent NaN
    for i, p in enumerate(predictions):
        val = p.data if isinstance(p, Value) else float(p)
        if not (0.0 <= val <= 1.0):
            raise ValueError(
                f"bce: predictions[{i}] = {val:.4f} is outside [0, 1]. "
                f"BCE expects probabilities. Did you forget sigmoid?"
            )

    n     = len(predictions)
    total = Value(0.0)

    for p, y in zip(predictions, targets):
        p   = _ensure_value(p)
        y   = float(y)
        # y * log(p) + (1 - y) * log(1 - p)
        term = _safe_log(p, eps) * y + _safe_log_1minus(p, eps) * (1.0 - y)
        total = total + term

    return total * (-1.0 / n)


def hinge(predictions, targets) -> Value:
    """
    Hinge Loss (SVM objective):
        L = (1/n) * Σ max(0, 1 - yᵢ * predᵢ)

    Gradient:
        ∂L/∂predᵢ = -yᵢ/n   if yᵢ * predᵢ < 1   (margin violated)
                  =  0        if yᵢ * predᵢ ≥ 1   (margin satisfied)

    Derivation:
        Let m = 1 - y*pred.
        hinge(pred) = max(0, m) = ReLU(1 - y*pred)
        d/d(pred) max(0, m) = d/d(pred)[1 - y*pred] * 1{m>0}
                            = -y * 1{y*pred < 1}

    What hinge measures:
        Hinge only penalises predictions that violate the margin (y*pred < 1).
        Predictions that are correct AND confident (y*pred ≥ 1) incur zero loss.
        This is the "max-margin" property: once you're right enough, stop pushing.
        MSE and BCE always penalise even correct predictions slightly.

    Comparison to BCE at same wrong prediction (pred=0.1, y=+1):
        BCE  (as probability): -log(0.1) ≈ 2.3,  gradient ≈ -10
        Hinge (as raw score):  max(0, 1-0.1)=0.9, gradient = -1
        Hinge's gradient is smaller — it doesn't have BCE's urgency.
        Tradeoff: hinge gives sparse gradients (zero for correct predictions),
        BCE gives dense gradients (always nonzero, even for correct predictions).

    Args:
        predictions: list of Value (raw scores, unbounded)
        targets:     list of float in {-1, +1}
    Returns:
        scalar Value
    """
    _validate_lengths(predictions, targets, 'hinge')

    for i, y in enumerate(targets):
        if float(y) not in (-1.0, 1.0):
            raise ValueError(
                f"hinge: targets[{i}] = {y}. "
                f"Hinge loss requires labels in {{-1, +1}}."
            )

    n     = len(predictions)
    total = Value(0.0)

    for p, y in zip(predictions, targets):
        total = total + _hinge_term(p, float(y))

    return total * (1.0 / n)


# ═══════════════════════════════════════════════════════════
# GRADIENT CHECK
# ═══════════════════════════════════════════════════════════

def gradient_check(
    loss_fn,
    predictions,
    targets,
    h: float = 1e-5,
    tol: float = 1e-3,
) -> list:
    """
    Verify analytical gradients against central finite differences.

    Two completely independent computation paths:
        Analytical: build graph → forward → backward → read .grad
        Numerical:  perturb each input by ±h → (f(+h) - f(-h)) / 2h

    Fresh Value objects are used for both passes to prevent stale
    gradient contamination from a previous backward call.

    Why central differences (not forward)?
        Central: O(h²) error.  Forward: O(h) error.
        With h=1e-5, central gives ~1e-10 accuracy vs ~1e-5.

    Args:
        loss_fn:     callable(predictions, targets) → Value
        predictions: list of Value or float — the base point
        targets:     list of float
        h:           perturbation size (default 1e-5)
        tol:         maximum allowed relative error (default 1e-3)

    Returns:
        list of (analytical_grad, numerical_grad, relative_error) per pred

    Raises:
        AssertionError if any relative error exceeds tol
    """
    base = [
        p.data if isinstance(p, Value) else float(p)
        for p in predictions
    ]

    # ── Analytical pass ────────────────────────────────
    preds_a = [Value(v) for v in base]
    loss_a  = loss_fn(preds_a, targets)
    loss_a.backward()
    analytical = [p.grad for p in preds_a]

    # ── Numerical pass ─────────────────────────────────
    def perturbed(i: int, delta: float):
        return [
            Value(base[j] + (delta if j == i else 0.0))
            for j in range(len(base))
        ]

    numerical = []
    for i in range(len(base)):
        lp = loss_fn(perturbed(i, +h), targets).data
        lm = loss_fn(perturbed(i, -h), targets).data
        numerical.append((lp - lm) / (2.0 * h))

    # ── Compare ────────────────────────────────────────
    results = []
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom   = max(abs(a), abs(n), 1e-8)
        rel_err = abs(a - n) / denom
        results.append((a, n, rel_err))
        assert rel_err < tol, (
            f"gradient_check FAILED at predictions[{i}]: "
            f"analytical={a:.8f}, numerical={n:.8f}, "
            f"relative_error={rel_err:.2e} > tol={tol:.2e}"
        )

    return results