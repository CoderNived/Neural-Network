import math
from engine.value import Value

def mse(predictions, targets):
    """
    L = (1/n) * Σ (pred_i - target_i)²
    predictions: list of Value
    targets:     list of float or Value
    returns:     single scalar Value
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions, "
            f"{len(targets)} targets"
        )
    n = len(predictions)

    # Wrap targets if needed
    targets = [t if isinstance(t, Value) else Value(t) for t in targets]

    # Each term is a graph node — loss is the final scalar root
    total = Value(0.0)
    for p, t in zip(predictions, targets):
        diff = p + (t * -1)       # pred - target
        total = total + diff * diff

    return total * (1.0 / n)


def bce(predictions, targets, eps=1e-7):
    """
    L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    predictions: list of Value (should be sigmoid outputs, in (0,1))
    targets:     list of float — binary labels, 0 or 1
    returns:     single scalar Value

    eps: clamping value to prevent log(0).
    Cost: mathematically inexact at boundaries, but sigmoid can't
    actually reach 0 or 1 in finite precision, so this is safe.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions, "
            f"{len(targets)} targets"
        )
    n = len(predictions)

    total = Value(0.0)
    for p, y in zip(predictions, targets):
        # Clamp p.data for log stability — but keep p in the graph
        # We clamp the data directly before the log operation
        p_clamped = max(eps, min(1.0 - eps, p.data))

        # Build log(p) as a Value node with correct backward
        # log_p.backward needs dp, not dp_clamped — but at boundaries
        # the gradient is already saturated, so the distinction is moot
        log_p    = _log(p, eps)
        log_1mp  = _log_1minus(p, eps)

        # y*log(p) + (1-y)*log(1-p)
        term = log_p * y + log_1mp * (1.0 - y)
        total = total + term

    # negate and average
    return total * (-1.0 / n)


def _log(v, eps=1e-7):
    """
    log(v) as a Value node with correct backward.
    d/dv log(v) = 1/v
    Clamps v.data to prevent log(0), but gradient uses clamped value too
    to stay consistent.
    """
    v = v if isinstance(v, Value) else Value(v)
    clamped = max(eps, min(1.0 - eps, v.data))
    out = Value(math.log(clamped), _parents=(v,), _op='log')

    def _backward():
        v.grad += (1.0 / clamped) * out.grad

    out._backward = _backward
    return out


def _log_1minus(v, eps=1e-7):
    """
    log(1 - v) as a Value node.
    d/dv log(1-v) = -1/(1-v)
    """
    v = v if isinstance(v, Value) else Value(v)
    clamped = max(eps, min(1.0 - eps, v.data))
    out = Value(math.log(1.0 - clamped), _parents=(v,), _op='log1m')

    def _backward():
        v.grad += (-1.0 / (1.0 - clamped)) * out.grad

    out._backward = _backward
    return out


def gradient_check(loss_fn, predictions, targets, h=1e-5):
    """
    For each prediction p_i:
      numerical  = (loss(p_i+h) - loss(p_i-h)) / 2h
      analytical = p_i.grad after backward()

    Returns list of (analytical, numerical, relative_error) per prediction.
    Raises AssertionError if any relative error > 1e-3.
    """
    # Analytical pass
    preds = [Value(p.data if isinstance(p, Value) else p) for p in predictions]
    loss = loss_fn(preds, targets)
    loss.backward()
    analytical = [p.grad for p in preds]

    # Numerical pass
    numerical = []
    for i in range(len(predictions)):
        base = predictions[i].data if isinstance(predictions[i], Value) else predictions[i]

        p_plus  = [Value((predictions[j].data if isinstance(predictions[j], Value)
                          else predictions[j]) + (h if j == i else 0))
                   for j in range(len(predictions))]
        p_minus = [Value((predictions[j].data if isinstance(predictions[j], Value)
                          else predictions[j]) - (h if j == i else 0))
                   for j in range(len(predictions))]

        loss_plus  = loss_fn(p_plus,  targets).data
        loss_minus = loss_fn(p_minus, targets).data
        numerical.append((loss_plus - loss_minus) / (2 * h))

    results = []
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_err = abs(a - n) / denom
        results.append((a, n, rel_err))
        assert rel_err < 1e-3, (
            f"Gradient check failed at prediction {i}: "
            f"analytical={a:.6f}, numerical={n:.6f}, "
            f"relative_error={rel_err:.2e}"
        )

    return results