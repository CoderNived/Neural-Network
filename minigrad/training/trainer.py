"""
training/trainer.py
────────────────────
Trainer: wires model, optimizer, data, loss, and metrics into a
complete training loop with early stopping, LR scheduling, and
best-model checkpointing.

Mental model:
  The Trainer owns the loop. Everything else is injected.
  It does not know what the model is — only that it has
  .parameters(), .zero_grad(), and is callable.
  This is the same separation PyTorch's training loop follows.

Canonical order per batch:
  zero_grad → forward → loss → backward → step

Canonical order per epoch:
  train_batches → [val_pass] → [scheduler.step] → [early_stop check]
"""

import copy
import math


class StepLRScheduler:
    """
    Multiply lr by `gamma` every `step_size` epochs.
    Standard step decay: lr drops geometrically at fixed intervals.

    Example: lr=0.1, step_size=100, gamma=0.5
      Epoch 0-99:   lr=0.1
      Epoch 100-199: lr=0.05
      Epoch 200+:    lr=0.025
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.5):
        self.optimizer  = optimizer
        self.step_size  = step_size
        self.gamma      = gamma
        self._epoch     = 0

    def step(self):
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    @property
    def current_lr(self) -> float:
        return self.optimizer.lr


class ReduceOnPlateauScheduler:
    """
    Reduce lr by `factor` when val loss has not improved for
    `patience` epochs. Classic adaptive scheduler.

    When val loss is unavailable, falls back to train loss.
    """

    def __init__(self, optimizer, patience: int = 20,
                 factor: float = 0.5, min_lr: float = 1e-6):
        self.optimizer  = optimizer
        self.patience   = patience
        self.factor     = factor
        self.min_lr     = min_lr
        self._best      = math.inf
        self._wait      = 0

    def step(self, loss: float):
        if loss < self._best - 1e-6:
            self._best = loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self._wait = 0

    @property
    def current_lr(self) -> float:
        return self.optimizer.lr


class EarlyStopper:
    """
    Stops training when val loss has not improved for `patience` epochs.
    Returns True from .should_stop(loss) when patience is exhausted.
    """

    def __init__(self, patience: int = 30, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = math.inf
        self._wait      = 0
        self.stopped_at = None

    def should_stop(self, loss: float, epoch: int) -> bool:
        if loss < self._best - self.min_delta:
            self._best = loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped_at = epoch
                return True
        return False

    def reset(self):
        self._best      = math.inf
        self._wait      = 0
        self.stopped_at = None


class Checkpoint:
    """
    Stores the best model weights seen during training.

    .save(model, epoch, val_loss) — saves a deep copy of parameters
    .restore(model)               — writes saved weights back to model
    """

    def __init__(self):
        self.best_loss   = math.inf
        self.best_epoch  = None
        self._state      = None   # list of (data, grad) tuples

    def save(self, model, epoch: int, loss: float):
        if loss < self.best_loss:
            self.best_loss  = loss
            self.best_epoch = epoch
            # Store (data, grad) for each parameter.
            # We don't deep-copy Value objects — just the scalar state.
            self._state = [(p.data, p.grad) for p in model.parameters()]

    def restore(self, model):
        if self._state is None:
            return
        params = model.parameters()
        if len(params) != len(self._state):
            raise ValueError(
                f"Checkpoint has {len(self._state)} params, "
                f"model has {len(params)}."
            )
        for param, (data, grad) in zip(params, self._state):
            param.data = data
            param.grad = grad

    @property
    def has_checkpoint(self) -> bool:
        return self._state is not None


class Trainer:
    """
    Full training loop with:
      - Mini-batch SGD via DataLoader
      - Optional validation pass each epoch
      - Learning rate scheduling (step or reduce-on-plateau)
      - Early stopping
      - Best-model checkpointing
      - Gradient monitoring (optional)
      - Structured logging

    Usage:
        trainer = Trainer(model, optimizer, loss_fn, train_loader,
                          val_loader=val_loader,
                          scheduler=StepLRScheduler(opt, 100, 0.5),
                          early_stopper=EarlyStopper(patience=50),
                          metrics=MetricsBundle(model.parameters()))
        history = trainer.fit(epochs=500)
    """

    def __init__(self, model, optimizer, loss_fn, train_loader,
                 val_loader=None,
                 scheduler=None,
                 early_stopper=None,
                 metrics=None,
                 print_every=50,
                 verbose=True):

        self.model         = model
        self.optimizer     = optimizer
        self.loss_fn       = loss_fn
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.scheduler     = scheduler
        self.early_stopper = early_stopper
        self.metrics       = metrics
        self.print_every   = print_every
        self.verbose       = verbose
        self.checkpoint    = Checkpoint()

        self._log: list[str] = []

    # ── Core loop ─────────────────────────────────────────────────

    def fit(self, epochs: int) -> dict:
        """
        Train for up to `epochs` epochs.

        Returns a history dict:
            {
              'train_loss': [...],
              'val_loss':   [...],
              'accuracy':   [...],
              'lr':         [...],
            }
        """
        history = {
            'train_loss': [],
            'val_loss':   [],
            'accuracy':   [],
            'lr':         [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch() if self.val_loader else None

            # ── Metrics ───────────────────────────────────────────
            if self.metrics:
                self.metrics.loss.record_train(train_loss)
                if val_loss is not None:
                    self.metrics.loss.record_val(val_loss)
                if self.metrics.grads:
                    self.metrics.grads.commit_epoch()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(
                self.metrics.accuracy.current if self.metrics else None
            )
            current_lr = self.optimizer.lr
            history['lr'].append(current_lr)

            # ── Checkpoint ────────────────────────────────────────
            monitor_loss = val_loss if val_loss is not None else train_loss
            self.checkpoint.save(self.model, epoch, monitor_loss)

            # ── Scheduler ─────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceOnPlateauScheduler):
                    self.scheduler.step(monitor_loss)
                else:
                    self.scheduler.step()

            # ── Early stopping ────────────────────────────────────
            if self.early_stopper is not None:
                if self.early_stopper.should_stop(monitor_loss, epoch):
                    self._log_line(
                        f"  [Early stop] epoch={epoch}, "
                        f"best_loss={self.early_stopper._best:.6f}"
                    )
                    break

            # ── Convergence / divergence ──────────────────────────
            if self.metrics:
                if self.metrics.loss.diverged():
                    self._log_line(f"  [DIVERGED] epoch={epoch}, loss={train_loss:.4f}")
                    break

            # ── Logging ───────────────────────────────────────────
            if self.verbose and epoch % self.print_every == 0:
                self._print_epoch(epoch, train_loss, val_loss, current_lr)

        return history

    # ── Train pass ────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        total_loss  = 0.0
        n_batches   = 0

        for batch_X, batch_y in self.train_loader:
            batch_loss = self._train_batch(batch_X, batch_y)
            total_loss += batch_loss
            n_batches  += 1

            # Accuracy tracking (per sample inside batch)
            if self.metrics and self.metrics.accuracy:
                for xi, yi in zip(batch_X, batch_y):
                    pred = self.model(xi)
                    pred_val = pred.data if hasattr(pred, 'data') else pred
                    self.metrics.accuracy.update(pred_val, yi)

        if self.metrics and self.metrics.accuracy:
            self.metrics.accuracy.commit_epoch()

        return total_loss / max(n_batches, 1)

    def _train_batch(self, batch_X, batch_y) -> float:
        """
        Process one mini-batch.
        For batch_size=1 this is identical to online SGD.
        For batch_size>1 we accumulate loss across the batch then step.
        """
        self.optimizer.zero_grad()

        # Forward + loss for entire batch
        preds = [self.model(xi) for xi in batch_X]

        # Wrap single-value outputs in list for loss functions
        if not isinstance(preds[0], list):
            preds_list = preds
        else:
            preds_list = [p[0] for p in preds]

        loss = self.loss_fn(preds_list, batch_y)

        # Backward + step
        loss.backward()

        # Gradient monitor
        if self.metrics and self.metrics.grads:
            self.metrics.grads.update()

        self.optimizer.step()

        return loss.data

    # ── Validation pass ───────────────────────────────────────────

    def _val_epoch(self) -> float:
        total_loss = 0.0
        n_batches  = 0
        for batch_X, batch_y in self.val_loader:
            preds = [self.model(xi) for xi in batch_X]
            if not isinstance(preds[0], list):
                preds_list = preds
            else:
                preds_list = [p[0] for p in preds]
            loss = self.loss_fn(preds_list, batch_y)
            total_loss += loss.data
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(self, dataset) -> dict:
        """
        Run inference on a dataset, return predictions and metrics.
        Does not affect model weights or gradients.
        """
        preds, targets = [], []
        for xi, yi in dataset:
            out = self.model(xi)
            val = out.data if hasattr(out, 'data') else float(out)
            preds.append(val)
            targets.append(float(yi))

        loss_val = self.loss_fn(
            [__import__('engine.value', fromlist=['Value']).Value(p) for p in preds],
            targets
        ).data

        # Accuracy (binary only)
        if all(t in (0.0, 1.0) for t in targets):
            correct = sum(
                1 for p, t in zip(preds, targets)
                if (round(p) == t)
            )
            acc = correct / len(targets)
        else:
            acc = None

        return {
            'predictions': preds,
            'targets':     targets,
            'loss':        loss_val,
            'accuracy':    acc,
        }

    # ── Logging ───────────────────────────────────────────────────

    def _print_epoch(self, epoch, train_loss, val_loss, lr):
        parts = [f"Epoch {epoch:>5}", f"train_loss={train_loss:.6f}"]
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")
        if self.metrics and self.metrics.accuracy.history:
            parts.append(f"acc={self.metrics.accuracy.current*100:.1f}%")
        parts.append(f"lr={lr:.5f}")
        line = " | ".join(parts)
        self._log_line(line)

    def _log_line(self, line: str):
        self._log.append(line)
        if self.verbose:
            print(line)

    def print_log(self):
        print("\n".join(self._log))