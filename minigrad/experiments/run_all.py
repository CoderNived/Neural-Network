"""
experiments/run_all.py
───────────────────────
Master runner. Executes every experiment in order:

  1. AND  — single neuron, linearly separable baseline
  2. OR   — single neuron, faster convergence sanity check
  3. XOR (single neuron) — expected failure, documents plateau
  4. XOR (MLP)           — shows hidden layer solving non-linear problem
  5. Regression          — MLP fitting sin(2πx), train/val split
  6. Failure modes       — vanishing, dead ReLU, exploding gradients

Run with:
    cd minigrad
    python experiments/run_all.py

Output is structured plain text. No dependencies beyond the stdlib.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math

from engine.value import Value
from engine.optimizer import SGD
from nn.network import MLP
from losses.losses import mse
from training.dataloader import Dataset, DataLoader
from training.metrics import MetricsBundle
from training.trainer import Trainer, StepLRScheduler, ReduceOnPlateauScheduler, EarlyStopper
from training.tasks import get_task
from training.failure_modes import run_all_failures


# ─────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────

def banner(title: str):
    width = 60
    print("\n" + "█" * width)
    pad = (width - len(title) - 2) // 2
    print("█" + " " * pad + title + " " * (width - pad - len(title) - 2) + "█")
    print("█" * width)

def section(label: str):
    print(f"\n  ┌─ {label}")

def result_line(label: str, value: str, status: str = ""):
    status_str = f"  [{status}]" if status else ""
    print(f"  │  {label:<22} {value}{status_str}")

def print_predictions(model, X, y, threshold=0.5):
    print("  │")
    print("  │  Input      Target   Pred     Result")
    print("  │  " + "─" * 42)
    for xi, yi in zip(X, y):
        pred = model(xi)
        pred_val = pred.data if hasattr(pred, 'data') else float(pred)
        predicted_class = 1.0 if pred_val >= threshold else 0.0
        ok = "✓" if predicted_class == float(yi) else "✗"
        print(f"  │  {str(xi):<12} {yi:<9} {pred_val:<9.4f} {ok}")

def print_regression_predictions(model, X, y):
    print("  │")
    print("  │  x        Target    Pred      Error")
    print("  │  " + "─" * 42)
    for xi, yi in zip(X, y):
        pred_val = model(xi).data
        err = abs(pred_val - yi)
        print(f"  │  {xi[0]:<9.3f} {yi:<10.4f} {pred_val:<10.4f} {err:.4f}")


# ─────────────────────────────────────────
# EXPERIMENT BUILDERS
# ─────────────────────────────────────────

def run_task_experiment(task_name, epochs=2000, lr=0.5,
                        batch_size=None, scheduler_type=None,
                        early_stop=True, seed=42):
    """
    Generic experiment runner for any TaskSpec.
    Returns the final result dict.
    """
    random.seed(seed)
    task = get_task(task_name)

    banner(f"EXPERIMENT: {task.name}")
    print(f"\n  {task.description}")

    # ── Build model ───────────────────────────────────────────────
    section("Architecture")
    arch = task.recommended_arch
    model = MLP(**arch)
    result_line("Model", str(model))
    result_line("Parameters", str(model.n_parameters()))

    # ── Build training components ─────────────────────────────────
    section("Training setup")
    train_dataset = task.dataset
    val_dataset   = task.val_dataset

    effective_batch = batch_size or len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) \
                   if val_dataset else None

    opt = SGD(model.parameters(), lr=lr)

    scheduler = None
    if scheduler_type == 'step':
        scheduler = StepLRScheduler(opt, step_size=epochs // 5, gamma=0.5)
        result_line("Scheduler", f"StepLR(step={epochs//5}, γ=0.5)")
    elif scheduler_type == 'plateau':
        scheduler = ReduceOnPlateauScheduler(opt, patience=50, factor=0.5)
        result_line("Scheduler", "ReduceOnPlateau(patience=50)")

    stopper = EarlyStopper(patience=100, min_delta=1e-5) if early_stop else None

    is_classification = task.metric_threshold is not None
    metrics = MetricsBundle(
        parameters=model.parameters(),
        accuracy_threshold=task.metric_threshold or 0.5,
        track_gradients=True,
    )

    result_line("Loss fn",    task.loss_fn.__name__)
    result_line("LR",         str(lr))
    result_line("Epochs",     str(epochs))
    result_line("Batch size", str(effective_batch))
    result_line("Early stop", str(early_stop))

    # ── Train ─────────────────────────────────────────────────────
    section("Training")
    trainer = Trainer(
        model, opt, task.loss_fn,
        train_loader, val_loader=val_loader,
        scheduler=scheduler, early_stopper=stopper,
        metrics=metrics,
        print_every=max(1, epochs // 10),
        verbose=True,
    )
    history = trainer.fit(epochs)

    # ── Results ───────────────────────────────────────────────────
    section("Results")
    final_train = history['train_loss'][-1]
    final_epoch = len(history['train_loss'])
    converged   = metrics.loss.has_converged()
    diverged    = metrics.loss.diverged()

    result_line("Final train loss", f"{final_train:.6f}",
                "CONVERGED" if converged else ("DIVERGED" if diverged else "RUNNING"))
    result_line("Epochs run", f"{final_epoch}/{epochs}")

    if history['val_loss'] and history['val_loss'][-1] is not None:
        result_line("Final val loss", f"{history['val_loss'][-1]:.6f}")
        result_line("Best val loss",
                    f"{metrics.loss.best_val:.6f} @ epoch {metrics.loss.best_epoch}")

    if is_classification and metrics.accuracy.history:
        result_line("Final accuracy",
                    f"{metrics.accuracy.current * 100:.1f}%",
                    "PASS" if metrics.accuracy.current >= 1.0 else "FAIL")

    if stopper and stopper.stopped_at:
        result_line("Stopped early", f"epoch {stopper.stopped_at}")

    # ── Predictions ───────────────────────────────────────────────
    section("Predictions")
    X = task.dataset.X
    y = task.dataset.y

    if is_classification:
        print_predictions(model, X, y, threshold=task.metric_threshold)
    else:
        print_regression_predictions(model, X, y)

    # Restore best checkpoint
    if trainer.checkpoint.has_checkpoint:
        trainer.checkpoint.restore(model)
        result_line("Checkpoint restored", f"epoch {trainer.checkpoint.best_epoch}")

    print()
    return {
        'task':     task_name,
        'model':    model,
        'history':  history,
        'metrics':  metrics,
        'converged': converged,
        'final_loss': final_train,
    }


# ─────────────────────────────────────────
# MINI-BATCH DEMONSTRATION
# ─────────────────────────────────────────

def run_minibatch_comparison(seed=42):
    """
    Compare online SGD (batch=1) vs mini-batch (batch=4) vs full-batch
    on the XOR task. Demonstrates why batch size affects loss curve shape.
    """
    banner("MINI-BATCH COMPARISON: XOR")
    print("\n  Online SGD vs mini-batch vs full-batch")
    print("  Same model, same lr, same epochs. Different gradient noise.")

    random.seed(seed)
    task = get_task('xor')
    epochs = 1000

    configs = [
        ('Online SGD (batch=1)',  1),
        ('Mini-batch (batch=2)',  2),
        ('Full-batch (batch=4)',  4),
    ]

    results = {}
    for name, bs in configs:
        random.seed(seed)
        model = MLP(**task.recommended_arch)
        opt   = SGD(model.parameters(), lr=0.5)
        loader = DataLoader(task.dataset, batch_size=bs, shuffle=True)
        trainer = Trainer(model, opt, task.loss_fn, loader,
                          print_every=epochs + 1, verbose=False)
        history = trainer.fit(epochs)
        final = history['train_loss'][-1]
        print(f"  {name:<30} final loss = {final:.6f}")
        results[name] = final

    print()
    return results


# ─────────────────────────────────────────
# LR SCHEDULE DEMONSTRATION
# ─────────────────────────────────────────

def run_lr_schedule_comparison(seed=42):
    """
    Compare fixed LR vs step decay vs reduce-on-plateau on XOR MLP.
    """
    banner("LR SCHEDULE COMPARISON: XOR")
    print("\n  Fixed LR vs step decay vs reduce-on-plateau")

    epochs = 2000
    task   = get_task('xor')

    configs = [
        ('Fixed LR=0.5',           'none'),
        ('Step decay (÷2 each 400)', 'step'),
        ('Reduce on plateau',        'plateau'),
    ]

    for name, sched_type in configs:
        random.seed(seed)
        model = MLP(**task.recommended_arch)
        opt   = SGD(model.parameters(), lr=0.5)
        loader = DataLoader(task.dataset, batch_size=len(task.dataset), shuffle=False)

        if sched_type == 'step':
            sched = StepLRScheduler(opt, step_size=400, gamma=0.5)
        elif sched_type == 'plateau':
            sched = ReduceOnPlateauScheduler(opt, patience=80, factor=0.5)
        else:
            sched = None

        trainer = Trainer(model, opt, task.loss_fn, loader,
                          scheduler=sched, print_every=epochs + 1, verbose=False)
        history = trainer.fit(epochs)
        final_loss = history['train_loss'][-1]
        final_lr   = history['lr'][-1]
        print(f"  {name:<35} loss={final_loss:.6f}  final_lr={final_lr:.5f}")

    print()


# ─────────────────────────────────────────
# GRADIENT MONITOR DEMO
# ─────────────────────────────────────────

def run_gradient_monitor_demo(seed=42):
    """
    Shows gradient statistics per layer during XOR training.
    Demonstrates that a healthy network has similar gradient
    magnitudes across layers.
    """
    banner("GRADIENT MONITOR: XOR MLP")
    print("\n  Per-layer gradient magnitude over training")
    print("  A healthy network: roughly equal across all layers")
    print("  Vanishing: layers near input have near-zero gradients\n")

    random.seed(seed)
    task  = get_task('xor')
    model = MLP(**task.recommended_arch)
    opt   = SGD(model.parameters(), lr=0.5)
    loader = DataLoader(task.dataset, batch_size=len(task.dataset), shuffle=False)

    metrics = MetricsBundle(parameters=model.parameters(), track_gradients=True)
    trainer = Trainer(model, opt, task.loss_fn, loader,
                      metrics=metrics, print_every=500, verbose=True)
    trainer.fit(500)

    section("Final gradient statistics per layer")
    for i, layer in enumerate(model.layers):
        grads = [abs(p.grad) for p in layer.parameters()]
        if grads:
            mean_g = sum(grads) / len(grads)
            max_g  = max(grads)
            status = ""
            if mean_g < 1e-5:
                status = " ← VANISHING"
            elif mean_g > 10:
                status = " ← EXPLODING"
            print(f"  │  Layer {i} ({layer._activation_name}): "
                  f"mean={mean_g:.2e}  max={max_g:.2e}{status}")
    print()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "MINIGRAD — RUN ALL" + " " * 24 + "█")
    print("█" + " " * 12 + "Neural Network from Scratch" + " " * 19 + "█")
    print("█" * 60)

    all_results = {}

    # ── 1. AND ────────────────────────────────────────────────────
    all_results['and'] = run_task_experiment(
        'and', epochs=2000, lr=0.5, early_stop=True
    )

    # ── 2. OR ─────────────────────────────────────────────────────
    all_results['or'] = run_task_experiment(
        'or', epochs=2000, lr=0.5, early_stop=True
    )

    # ── 3. XOR — single neuron (expected failure) ─────────────────
    all_results['xor_neuron'] = run_task_experiment(
        'xor_neuron', epochs=3000, lr=0.5, early_stop=True
    )

    # ── 4. XOR — MLP ──────────────────────────────────────────────
    all_results['xor'] = run_task_experiment(
        'xor', epochs=3000, lr=0.5, early_stop=True,
        scheduler_type='plateau'
    )

    # ── 5. Regression ─────────────────────────────────────────────
    all_results['regression'] = run_task_experiment(
        'regression', epochs=3000, lr=0.05, early_stop=True,
        scheduler_type='step'
    )

    # ── 6. Mini-batch comparison ───────────────────────────────────
    run_minibatch_comparison()

    # ── 7. LR schedule comparison ─────────────────────────────────
    run_lr_schedule_comparison()

    # ── 8. Gradient monitor ────────────────────────────────────────
    run_gradient_monitor_demo()

    # ── 9. Failure modes ──────────────────────────────────────────
    banner("FAILURE MODES")
    run_all_failures()

    # ── Final summary ─────────────────────────────────────────────
    banner("FINAL SUMMARY")
    print()
    rows = [
        ('AND',              all_results['and']),
        ('OR',               all_results['or']),
        ('XOR (single)',     all_results['xor_neuron']),
        ('XOR (MLP)',        all_results['xor']),
        ('Regression',       all_results['regression']),
    ]
    print(f"  {'Task':<22} {'Final Loss':<14} {'Converged':<12} {'Status'}")
    print("  " + "─" * 56)
    for name, r in rows:
        loss = f"{r['final_loss']:.6f}"
        conv = str(r['converged'])
        # Pass = accuracy 100% OR loss below threshold OR converged
        acc = r['metrics'].accuracy.current if r['metrics'].accuracy.history else None
        passed = (r['converged'] or
                  (acc is not None and acc >= 1.0) or
                  r["final_loss"] < (0.2 if "Regression" in name else 0.05))
        status = "✓ PASS" if passed else "✗ FAIL"
        # XOR single neuron is expected to fail — plateau at 0.25 is the signal
        if 'single' in name.lower():
            status = "✓ EXPECTED FAIL" if abs(r['final_loss'] - 0.25) < 0.01 else "✗ UNEXPECTED"
        print(f"  {name:<22} {loss:<14} {conv:<12} {status}")
    print()


if __name__ == '__main__':
    main()