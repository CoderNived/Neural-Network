"""
training/metrics.py
────────────────────
Loss tracking, accuracy, convergence detection, and curve history.

Design principle: metrics are pure recorders. They consume numbers and
report statistics. They have no knowledge of Value objects, models, or
optimizers. This keeps them composable and testable independently.
"""

import math


class LossTracker:
    """
    Tracks per-epoch train (and optionally val) loss.
    Supports convergence detection and plateau detection.
    """

    def __init__(self):
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self._best_val    = math.inf
        self._best_epoch  = 0

    # ── Recording ─────────────────────────────────────────────────

    def record_train(self, loss: float):
        self.train_losses.append(float(loss))

    def record_val(self, loss: float):
        val = float(loss)
        self.val_losses.append(val)
        if val < self._best_val:
            self._best_val   = val
            self._best_epoch = len(self.val_losses) - 1

    @property
    def current_train(self) -> float:
        return self.train_losses[-1] if self.train_losses else math.inf

    @property
    def current_val(self) -> float:
        return self.val_losses[-1] if self.val_losses else math.inf

    @property
    def best_val(self) -> float:
        return self._best_val

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    # ── Convergence detection ──────────────────────────────────────

    def has_converged(self, window: int = 20, tol: float = 1e-5) -> bool:
        """
        Returns True if the training loss has not changed by more than
        `tol` (relative) over the last `window` epochs.

        Uses relative change to handle losses at very different scales.
        A network that drops from 0.001 to 0.0009999 is converged.
        One that drops from 10.0 to 9.0 is not.
        """
        if len(self.train_losses) < window:
            return False
        recent = self.train_losses[-window:]
        base   = abs(recent[0]) + 1e-10  # avoid division by zero
        change = abs(recent[-1] - recent[0]) / base
        return change < tol

    def is_plateau(self, window: int = 30, tol: float = 1e-4) -> bool:
        """
        Detects a loss plateau: variance of last `window` losses is
        below `tol`. Subtly different from convergence — plateau
        means the loss isn't moving, not necessarily that it's low.
        A stuck network (bad lr, dead neurons) produces a plateau at
        high loss. Convergence implies both plateau and low loss.
        """
        if len(self.train_losses) < window:
            return False
        recent   = self.train_losses[-window:]
        mean     = sum(recent) / window
        variance = sum((x - mean) ** 2 for x in recent) / window
        return variance < tol ** 2

    def diverged(self, threshold: float = 1e4) -> bool:
        """Loss has exploded above threshold."""
        return (self.train_losses and
                (math.isnan(self.train_losses[-1]) or
                 abs(self.train_losses[-1]) > threshold))

    # ── Smoothing ──────────────────────────────────────────────────

    def smoothed(self, alpha: float = 0.9) -> list[float]:
        """
        Exponential moving average of train loss.
        alpha=0.9: heavily smoothed (good for plotting).
        alpha=0.1: lightly smoothed (tracks raw loss closely).
        """
        if not self.train_losses:
            return []
        smoothed = [self.train_losses[0]]
        for loss in self.train_losses[1:]:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * loss)
        return smoothed

    def reset(self):
        self.train_losses = []
        self.val_losses   = []
        self._best_val    = math.inf
        self._best_epoch  = 0


class AccuracyTracker:
    """
    Tracks classification accuracy per epoch.
    threshold: decision boundary for binary classification (default 0.5).
    For hinge-loss classifiers, use threshold=0 (sign of score).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold  = threshold
        self.history:   list[float] = []
        self._correct   = 0
        self._total     = 0

    def update(self, pred: float, target: float):
        """Call once per sample within an epoch."""
        self._total += 1
        if self.threshold == 0:
            # Hinge-style: correct if sign matches
            predicted = 1.0 if pred >= 0 else -1.0
            correct   = 1.0 if target >= 0 else -1.0
            if predicted == correct:
                self._correct += 1
        else:
            predicted = 1.0 if pred >= self.threshold else 0.0
            if predicted == float(target):
                self._correct += 1

    def commit_epoch(self) -> float:
        """Finalize the epoch, record accuracy, reset counters."""
        acc = self._correct / self._total if self._total > 0 else 0.0
        self.history.append(acc)
        self._correct = 0
        self._total   = 0
        return acc

    @property
    def current(self) -> float:
        return self.history[-1] if self.history else 0.0

    def reset(self):
        self.history  = []
        self._correct = 0
        self._total   = 0


class GradientMonitor:
    """
    Monitors gradient statistics per parameter per epoch.
    Used to detect vanishing gradients, dead neurons, and exploding gradients.

    Attach to a model's parameters and call update() after each backward().
    """

    def __init__(self, parameters, names=None):
        self.parameters = list(parameters)
        self.names      = names or [f"p{i}" for i in range(len(self.parameters))]
        # History: list of dicts, one per epoch, {name: grad_value}
        self.history:   list[dict] = []
        self._epoch_grads: dict    = {n: [] for n in self.names}

    def update(self):
        """Call after each backward() within an epoch."""
        for name, param in zip(self.names, self.parameters):
            self._epoch_grads[name].append(abs(param.grad))

    def commit_epoch(self) -> dict:
        """
        Compute per-parameter mean |grad| for the epoch.
        Returns {name: mean_abs_grad}.
        """
        snapshot = {}
        for name in self.names:
            grads = self._epoch_grads[name]
            snapshot[name] = sum(grads) / len(grads) if grads else 0.0
            self._epoch_grads[name] = []
        self.history.append(snapshot)
        return snapshot

    def diagnose(self, epoch_snapshot: dict) -> list[str]:
        """
        Returns a list of human-readable diagnostic warnings.
        Thresholds are heuristic but calibrated for this scalar engine.
        """
        warnings = []
        for name, mean_grad in epoch_snapshot.items():
            if mean_grad < 1e-6:
                warnings.append(
                    f"[VANISHING] {name}: mean|grad|={mean_grad:.2e} — "
                    f"gradient effectively zero. "
                    f"Possible causes: sigmoid/tanh saturation in deep network."
                )
            elif mean_grad > 1e2:
                warnings.append(
                    f"[EXPLODING] {name}: mean|grad|={mean_grad:.2e} — "
                    f"gradient too large. "
                    f"Likely cause: lr too high or no gradient clipping."
                )
        return warnings

    def dead_neuron_check(self, model) -> list[str]:
        """
        Check for dead ReLU neurons: neurons whose weights have never
        produced a positive pre-activation (so ReLU output is always 0,
        gradient is always 0).
        Heuristic: weight grad == 0.0 for all weights in a neuron.
        """
        warnings = []
        for layer_idx, layer in enumerate(getattr(model, 'layers', [])):
            for neuron_idx, neuron in enumerate(layer.neurons):
                if layer._activation_name != 'relu':
                    continue
                all_zero = all(abs(w.grad) < 1e-9 for w in neuron.w)
                if all_zero:
                    warnings.append(
                        f"[DEAD RELU] Layer {layer_idx}, Neuron {neuron_idx}: "
                        f"all weight grads ≈ 0. "
                        f"This neuron contributes nothing to the network."
                    )
        return warnings

    def reset(self):
        self.history = []
        self._epoch_grads = {n: [] for n in self.names}


class MetricsBundle:
    """
    Convenience wrapper that holds LossTracker, AccuracyTracker,
    and GradientMonitor together. Passed into Trainer.
    """

    def __init__(self, parameters=None, param_names=None,
                 accuracy_threshold=0.5, track_gradients=True):
        self.loss     = LossTracker()
        self.accuracy = AccuracyTracker(threshold=accuracy_threshold)
        self.grads    = (GradientMonitor(parameters, param_names)
                         if (track_gradients and parameters is not None)
                         else None)

    def reset(self):
        self.loss.reset()
        self.accuracy.reset()
        if self.grads:
            self.grads.reset()

    def summary(self) -> str:
        lines = [
            f"  Train loss : {self.loss.current_train:.6f}",
            f"  Val loss   : {self.loss.current_val:.6f}" if self.loss.val_losses else "",
            f"  Accuracy   : {self.accuracy.current * 100:.1f}%",
            f"  Converged  : {self.loss.has_converged()}",
            f"  Diverged   : {self.loss.diverged()}",
        ]
        return "\n".join(l for l in lines if l)