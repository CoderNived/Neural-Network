"""
training/failure_modes.py
──────────────────────────
Deliberately constructed failure scenarios with diagnostics.

Each failure mode:
  1. Builds a model and training setup designed to trigger the failure
  2. Trains for N steps
  3. Collects diagnostic evidence
  4. Prints a structured report explaining what happened and why

These are not "things that go wrong accidentally" — they are
controlled experiments. The goal is to observe the failure clearly,
measure it, and understand it mechanically.

Failure modes covered:
  1. Vanishing gradients  — deep sigmoid network (classic)
  2. Dead ReLU neurons    — ReLU with bad initialization or high lr
  3. Exploding gradients  — no clipping, high lr, deep network

Run each with:
    from training.failure_modes import run_all_failures
    run_all_failures()
"""

import random
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.value import Value
from engine.optimizer import SGD
from nn.network import MLP
from losses.losses import mse


# ─────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────

def _header(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

def _section(label: str):
    print(f"\n── {label} {'─' * (54 - len(label))}")

def _measure_grad_stats(model) -> dict:
    """Return mean and max |grad| per layer."""
    stats = {}
    for i, layer in enumerate(model.layers):
        grads = [abs(p.grad) for p in layer.parameters() if p.grad != 0.0]
        if grads:
            stats[f"layer_{i}"] = {
                'mean': sum(grads) / len(grads),
                'max':  max(grads),
                'min':  min(grads),
                'n_zero': sum(1 for p in layer.parameters() if abs(p.grad) < 1e-9),
                'n_total': len(layer.parameters()),
            }
        else:
            stats[f"layer_{i}"] = {
                'mean': 0.0, 'max': 0.0, 'min': 0.0,
                'n_zero': len(layer.parameters()),
                'n_total': len(layer.parameters()),
            }
    return stats

def _print_grad_stats(stats: dict):
    for layer_name, s in stats.items():
        dead_pct = s['n_zero'] / s['n_total'] * 100
        print(f"  {layer_name}: mean={s['mean']:.2e}  "
              f"max={s['max']:.2e}  "
              f"dead={s['n_zero']}/{s['n_total']} ({dead_pct:.0f}%)")

def _single_forward_backward(model, X, y, optimizer):
    """One full step: zero → forward → loss → backward → step."""
    optimizer.zero_grad()
    preds = [model(xi) for xi in X]
    p_list = [p if isinstance(p, list) else p for p in preds]
    # unwrap single-output
    if hasattr(p_list[0], 'data'):
        loss = mse(p_list, y)
    else:
        loss = mse([p[0] for p in p_list], y)
    loss.backward()
    optimizer.step()
    return loss.data


# ─────────────────────────────────────────
# FAILURE 1 — VANISHING GRADIENTS
# ─────────────────────────────────────────

def failure_vanishing_gradients(n_steps=100, verbose=True):
    """
    Trigger: deep sigmoid network (5 hidden layers).

    Why it happens:
      sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
      Maximum value is 0.25 (at x=0).
      In a chain of 5 sigmoid layers:
        grad at input ≈ (0.25)^5 = 0.00098

    Each layer multiplies the gradient by at most 0.25.
    By the time the gradient reaches the first layer,
    it is ~1000x smaller than at the output.
    The early layers learn almost nothing.

    Diagnosis:
      Layer 4 (output): grad ~ 0.01 (visible)
      Layer 0 (input):  grad ~ 1e-5 (effectively zero)
    """
    _header("FAILURE 1 — Vanishing Gradients (deep sigmoid)")

    random.seed(42)

    # 5 hidden sigmoid layers → classic vanishing gradient setup
    model = MLP(
        layer_sizes=[2, 4, 4, 4, 4, 4, 1],
        activations=['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
    )
    opt = SGD(model.parameters(), lr=0.1)

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]

    _section("Architecture")
    print(f"  {model}")
    print(f"  Depth: {len(model.layers)} layers of sigmoid")
    print(f"  Theoretical max grad per layer: 0.25")
    print(f"  Theoretical max grad at layer 0: 0.25^6 ≈ {0.25**6:.2e}")

    _section("Gradient magnitude by layer (after 1 step)")
    _single_forward_backward(model, X, y, opt)
    stats = _measure_grad_stats(model)
    _print_grad_stats(stats)

    _section(f"Training for {n_steps} steps")
    losses = []
    for step in range(n_steps):
        loss = _single_forward_backward(model, X, y, opt)
        losses.append(loss)

    # Re-measure after training
    _single_forward_backward(model, X, y, opt)
    stats_after = _measure_grad_stats(model)

    _section("Gradient magnitude after training")
    _print_grad_stats(stats_after)

    _section("Loss trajectory")
    checkpoints = [0, 10, 25, 50, 99]
    for i in checkpoints:
        if i < len(losses):
            print(f"  step {i:>4}: loss={losses[i]:.6f}")

    _section("Diagnosis")
    layer_means = [stats_after.get(f'layer_{i}', {}).get('mean', 0)
                   for i in range(len(model.layers))]

    if layer_means and layer_means[0] < layer_means[-1] * 0.01:
        ratio = layer_means[-1] / (layer_means[0] + 1e-12)
        print(f"  ✗ VANISHING GRADIENT CONFIRMED")
        print(f"    Gradient ratio (output/input layers): {ratio:.1f}x")
        print(f"    First layer mean|grad| = {layer_means[0]:.2e}")
        print(f"    Last layer  mean|grad| = {layer_means[-1]:.2e}")
    else:
        print(f"  Gradients appear relatively uniform across layers.")

    _section("Fix")
    print("  1. Replace sigmoid with ReLU (gradient doesn't decay in positive region)")
    print("  2. Use tanh for shallow networks (max grad = 1.0 vs 0.25)")
    print("  3. Residual connections (skip connections bypass the chain)")
    print("  4. Batch normalization (re-centers activations, keeps grads healthy)")
    print("  5. Careful weight initialization (Xavier/He scale by fan-in)")

    return {'losses': losses, 'grad_stats_after': stats_after}


# ─────────────────────────────────────────
# FAILURE 2 — DEAD ReLU
# ─────────────────────────────────────────

def failure_dead_relu(n_steps=200, verbose=True):
    """
    Trigger: large negative bias initialization or high learning rate
    causes ReLU inputs to be permanently negative → output always 0
    → gradient always 0 → neuron never recovers.

    Why it happens:
      relu(x) = max(0, x)
      relu'(x) = 1 if x > 0, else 0

    Once the pre-activation (w·x + b) is always negative for all
    training samples, the gradient through this neuron is exactly 0.
    The optimizer receives no signal to fix it.
    The neuron is permanently dead.

    We trigger this by initializing biases to large negative values.
    """
    _header("FAILURE 2 — Dead ReLU Neurons")

    random.seed(42)

    model = MLP(
        layer_sizes=[2, 8, 1],
        activations=['relu', 'linear']
    )
    opt = SGD(model.parameters(), lr=0.05)

    # Force dead neurons: set hidden layer biases to large negative
    _section("Triggering dead neurons")
    print("  Setting hidden layer biases to -5.0 (pre-activation always negative)")
    for neuron in model.layers[0].neurons:
        neuron.b.data = -5.0

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]

    # Check dead neurons before training
    _section("Dead neuron count before training")
    _single_forward_backward(model, X, y, opt)
    hidden_layer = model.layers[0]
    dead_before = sum(
        1 for n in hidden_layer.neurons
        if all(abs(w.grad) < 1e-9 for w in n.w)
    )
    print(f"  Dead neurons in hidden layer: {dead_before}/{len(hidden_layer.neurons)}")

    _section(f"Training for {n_steps} steps")
    losses = []
    for step in range(n_steps):
        loss = _single_forward_backward(model, X, y, opt)
        losses.append(loss)

    _section("Dead neuron count after training")
    _single_forward_backward(model, X, y, opt)
    dead_after = sum(
        1 for n in hidden_layer.neurons
        if all(abs(w.grad) < 1e-9 for w in n.w)
    )
    print(f"  Dead neurons in hidden layer: {dead_after}/{len(hidden_layer.neurons)}")

    _section("Loss trajectory")
    checkpoints = [0, 10, 50, 100, 199]
    for i in checkpoints:
        if i < len(losses):
            print(f"  step {i:>4}: loss={losses[i]:.6f}")

    _section("Diagnosis")
    if dead_after > 0:
        print(f"  ✗ DEAD RELU CONFIRMED: {dead_after} neurons permanently inactive")
        print(f"    These neurons contribute zero gradient to all layers before them.")
        print(f"    Effective network width is {len(hidden_layer.neurons) - dead_after}"
              f"/{len(hidden_layer.neurons)}")
    else:
        print("  All neurons recovered (rare — usually requires careful lr tuning)")

    _section("Gradient stats")
    stats = _measure_grad_stats(model)
    _print_grad_stats(stats)

    _section("Fix")
    print("  1. Leaky ReLU: relu(x) = max(0.01x, x) — small gradient even for x<0")
    print("  2. He initialization: std = sqrt(2/fan_in) — keeps pre-activations positive")
    print("  3. Careful bias init: start at 0.0, not negative values")
    print("  4. Lower learning rate: large steps can push neurons into dead zone")
    print("  5. Monitor % dead neurons during training as a health metric")

    return {'losses': losses, 'dead_before': dead_before, 'dead_after': dead_after}


# ─────────────────────────────────────────
# FAILURE 3 — EXPLODING GRADIENTS
# ─────────────────────────────────────────

def failure_exploding_gradients(n_steps=30, verbose=True):
    """
    Trigger: high learning rate + deep network + linear activations.

    Why it happens:
      In a chain of matrix multiplications with large weights,
      gradients multiply at each layer (chain rule).
      With no activation saturation to bound them, they grow
      exponentially with depth.

      With lr=10 and a 4-layer linear network:
        grad at output: O(1)
        grad at layer 0: O(W^4) — can be enormous

    The optimizer then takes a massive step, overshoots badly,
    and the loss explodes to NaN or thousands.
    """
    _header("FAILURE 3 — Exploding Gradients")

    random.seed(42)

    # Linear activations: no saturation, gradients can grow unboundedly
    model = MLP(
        layer_sizes=[2, 4, 4, 4, 1],
        activations=['linear', 'linear', 'linear', 'linear']
    )
    opt = SGD(model.parameters(), lr=10.0)   # deliberately too high

    # Initialize weights to slightly large values to amplify explosion
    for layer in model.layers:
        for neuron in layer.neurons:
            for w in neuron.w:
                w.data *= 3.0

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]

    _section("Architecture + setup")
    print(f"  {model}")
    print(f"  Activations: all linear (no saturation)")
    print(f"  Learning rate: {opt.lr}")
    print(f"  Initial weights: scaled 3x")

    _section(f"Training for {n_steps} steps")
    losses = []
    exploded_at = None
    for step in range(n_steps):
        try:
            loss = _single_forward_backward(model, X, y, opt)
            losses.append(loss)
            if math.isnan(loss) or abs(loss) > 1e6:
                exploded_at = step
                if verbose:
                    print(f"  step {step:>4}: loss={loss} ← EXPLODED")
                break
            if verbose and (step < 5 or step % 5 == 0):
                print(f"  step {step:>4}: loss={loss:.4f}")
        except (OverflowError, ValueError) as e:
            exploded_at = step
            print(f"  step {step:>4}: {type(e).__name__} — {e}")
            break

    _section("Gradient stats at explosion point")
    try:
        _single_forward_backward(model, X, y, opt)
        stats = _measure_grad_stats(model)
        _print_grad_stats(stats)
    except Exception as e:
        print(f"  Cannot measure grads — model state corrupted: {e}")

    _section("Diagnosis")
    if exploded_at is not None:
        print(f"  ✗ EXPLODING GRADIENT CONFIRMED: loss diverged at step {exploded_at}")
    elif losses and abs(losses[-1]) > abs(losses[0]) * 10:
        print(f"  ✗ GRADIENT EXPLOSION: loss grew {abs(losses[-1])/abs(losses[0]+1e-10):.1f}x")
    else:
        print(f"  Loss did not explode in {n_steps} steps with this seed.")

    _section("Fix")
    print("  1. Gradient clipping: p.grad = clip(p.grad, -c, c) before step()")
    print("     Standard threshold: c=1.0 (Pascanu et al., 2013)")
    print("  2. Lower learning rate: most direct fix")
    print("  3. Weight regularization: L2 penalty keeps weights bounded")
    print("  4. Saturating activations (tanh, sigmoid): bound the gradient")
    print("  5. Batch normalization: normalizes pre-activations each layer")

    return {'losses': losses, 'exploded_at': exploded_at}


# ─────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────

def run_all_failures():
    results = {}

    results['vanishing'] = failure_vanishing_gradients()
    results['dead_relu'] = failure_dead_relu()
    results['exploding'] = failure_exploding_gradients()

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  Vanishing: final loss = {results['vanishing']['losses'][-1]:.4f}")
    print(f"  Dead ReLU: {results['dead_relu']['dead_after']} dead neurons after training")
    exploded = results['exploding']['exploded_at']
    print(f"  Exploding: diverged at step {exploded}" if exploded else
          f"  Exploding: loss = {results['exploding']['losses'][-1]:.4f}")
    print()

    return results


if __name__ == '__main__':
    run_all_failures()