"""
training/optimizers.py
----------------------
All optimizers for minigrad in one place.

Design contract
---------------
Every optimizer must subclass Optimizer and implement:
    step()      — update every parameter by one step
    zero_grad() — reset all parameter gradients to 0.0

No optimizer may touch any attribute of a Value except .data and .grad.
No optimizer may import anything outside the standard library.

Optimizer inventory
-------------------
    SGD           — vanilla stochastic gradient descent
    SGDMomentum   — SGD + exponential moving average of gradients
    SGDStepDecay  — SGD with multiplicative LR decay at fixed intervals
    Adam          — adaptive moment estimation (Kingma & Ba, 2014)

Choosing an optimizer
---------------------
    XOR / tiny networks    → SGD at LR 0.5–3.0  (direct, predictable)
    Same + faster          → SGDMomentum β=0.9, LR 0.05–0.5
    Larger networks        → Adam α=0.001 (default), β₁=0.9, β₂=0.999

Learning rate guidance
----------------------
Adam's effective step size is already normalised by the second moment,
so its LR is NOT interchangeable with SGD's.  Using SGD's optimal LR
(e.g. 2.807) with Adam will diverge immediately.  Adam's canonical
default is α = 0.001.

Step decay guidance
-------------------
The convergence zone for XOR is roughly [0.30, 2.81].  With base LR
2.807 and decay factor 0.5, the LR crosses below 0.30 after:
    step 1: 2.807 * 0.5   = 1.404   (ok)
    step 2: 1.404 * 0.5   = 0.702   (ok)
    step 3: 0.702 * 0.5   = 0.351   (ok)
    step 4: 0.351 * 0.5   = 0.175   (BELOW threshold — dead)
So with step_size=50 epochs the network stalls at epoch 200.  Set
step_size large enough that meaningful descent happens before each decay.
A safer default: step_size=100, factor=0.5 → stays in zone for 300 epochs.
"""

import math


# ═══════════════════════════════════════════════════════════
# BASE CLASS — the contract every optimizer must honour
# ═══════════════════════════════════════════════════════════

class Optimizer:
    """
    Abstract base class.  Subclasses must override step() and zero_grad().
    Storing the parameter list here avoids repeating it in every subclass.
    """

    def __init__(self, params):
        """
        params — list of Value objects (model.parameters())
        Validates on construction so failures surface immediately,
        not silently on the first training step.
        """
        if not params:
            raise ValueError(
                "Optimizer received an empty parameter list. "
                "Did you call model.parameters() before constructing the optimizer?"
            )
        self.params = list(params)

    def step(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement step()"
        )

    def zero_grad(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement zero_grad()"
        )

    # ── shared helper ────────────────────────────────────
    def _zero_all(self):
        """Reset .grad on every parameter.  Called by every zero_grad()."""
        for p in self.params:
            p.grad = 0.0


# ═══════════════════════════════════════════════════════════
# SGD — vanilla gradient descent
# ═══════════════════════════════════════════════════════════

class SGD(Optimizer):
    """
    θ ← θ - lr * ∂L/∂θ

    The simplest possible optimizer.  No state beyond the LR.
    Works well on XOR with LR in [0.30, 2.81].

    Args:
        params      list of Value objects
        lr          learning rate (default 0.1)
        weight_decay  L2 regularisation coefficient, applied to weights
                      only — never pass bias parameters here if you want
                      unregularised biases (default 0.0 = off)
    """

    def __init__(self, params, lr=0.1, weight_decay=0.0):
        super().__init__(params)
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.lr           = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data   # L2 penalty
            p.data -= self.lr * grad

    def zero_grad(self):
        self._zero_all()


# ═══════════════════════════════════════════════════════════
# SGD WITH MOMENTUM
# ═══════════════════════════════════════════════════════════

class SGDMomentum(Optimizer):
    """
    Velocity update:   v ← β·v + g          (g = gradient)
    Weight update:     θ ← θ - lr·v

    At steady state with constant gradient g:
        v* = g / (1 - β)
    So momentum amplifies the effective step by 1/(1-β).
    With β=0.9 this is a 10× amplification of a consistent gradient,
    which is why momentum converges faster on smooth loss surfaces.

    Velocity vectors are initialised to zero.  Do NOT reuse an
    SGDMomentum instance across separate training runs — the stale
    velocity from run 1 will corrupt the first steps of run 2.

    Args:
        params      list of Value objects
        lr          learning rate (default 0.01)
        beta        momentum coefficient, 0 < β < 1 (default 0.9)
        weight_decay  L2 regularisation (default 0.0)
    """

    def __init__(self, params, lr=0.01, beta=0.9, weight_decay=0.0):
        super().__init__(params)
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.lr           = lr
        self.beta         = beta
        self.weight_decay = weight_decay
        # velocity per parameter, initialised to zero
        self.velocity     = [0.0] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.velocity[i] = self.beta * self.velocity[i] + grad
            p.data -= self.lr * self.velocity[i]

    def zero_grad(self):
        self._zero_all()
        # NOTE: velocity is NOT reset here.  Gradients and velocity are
        # separate concepts.  Resetting velocity is the caller's choice
        # (e.g. when starting a new training run), not automatic.


# ═══════════════════════════════════════════════════════════
# SGD WITH STEP DECAY
# ═══════════════════════════════════════════════════════════

class SGDStepDecay(Optimizer):
    """
    Vanilla SGD whose LR is multiplied by `factor` every `step_size` epochs.

    The caller must call .scheduler_step(epoch) once per epoch AFTER
    the parameter update.  This keeps scheduling logic explicit and
    avoids hidden coupling between the optimizer and the training loop.

    Choosing step_size and factor:
        With base LR 2.807 and convergence floor 0.30:
            max safe decays = floor(log(0.30/2.807) / log(factor))
        For factor=0.5: 4 decays before death.
        With step_size=100: first decay at epoch 100 → safe for 400 epochs.

        Rule of thumb: set step_size so at least 2 full convergence
        phases happen before the first decay.

    Args:
        params      list of Value objects
        lr          initial learning rate
        step_size   decay every this many epochs (default 100)
        factor      multiplicative decay factor, 0 < factor < 1 (default 0.5)
        weight_decay  L2 regularisation (default 0.0)
        min_lr      floor — LR never drops below this (default 1e-6)
    """

    def __init__(self, params, lr=0.1, step_size=100, factor=0.5,
                 weight_decay=0.0, min_lr=1e-6):
        super().__init__(params)
        if not (0.0 < factor < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        self.lr           = lr
        self._base_lr     = lr      # kept for logging / reset
        self.step_size    = step_size
        self.factor       = factor
        self.weight_decay = weight_decay
        self.min_lr       = min_lr

    def step(self):
        for p in self.params:
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            p.data -= self.lr * grad

    def scheduler_step(self, epoch):
        """
        Call once per epoch.  Decays LR when epoch is a multiple of step_size.
        Returns the current LR so the training loop can log it.
        """
        if epoch > 0 and epoch % self.step_size == 0:
            self.lr = max(self.lr * self.factor, self.min_lr)
        return self.lr

    def zero_grad(self):
        self._zero_all()


# ═══════════════════════════════════════════════════════════
# ADAM — adaptive moment estimation
# ═══════════════════════════════════════════════════════════

class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation  (Kingma & Ba, 2014)

    Maintains per-parameter first and second moment estimates:
        m ← β₁·m + (1 - β₁)·g           first moment  (mean of g)
        v ← β₂·v + (1 - β₂)·g²          second moment (variance of g)

    Bias correction (critical for early steps when m,v ≈ 0):
        m̂ = m / (1 - β₁ᵗ)
        v̂ = v / (1 - β₂ᵗ)

    Parameter update:
        θ ← θ - α · m̂ / (√v̂ + ε)

    Why this beats SGD on sparse/noisy gradients:
        Parameters with small, inconsistent gradients get a larger
        effective LR (small v̂ → large step).  Parameters with large,
        consistent gradients get a smaller effective LR (large v̂ →
        small step).  This is automatic per-parameter normalisation.

    IMPORTANT: Adam's LR is NOT the same as SGD's LR.
        SGD optimal for XOR: ~2.807
        Adam default:        0.001
        Using 2.807 with Adam will cause immediate divergence.

    Args:
        params   list of Value objects
        lr       learning rate α (default 0.001 — the canonical default)
        beta1    first moment decay,  0 < β₁ < 1 (default 0.9)
        beta2    second moment decay, 0 < β₂ < 1 (default 0.999)
        eps      numerical stability floor (default 1e-8)
        weight_decay  L2 regularisation, applied before moment update
                      (default 0.0)
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")

        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay

        # per-parameter state — all zero before the first step
        self._m  = [0.0] * len(self.params)   # first moment
        self._v  = [0.0] * len(self.params)   # second moment
        self._t  = 0                           # step counter (shared)

    def step(self):
        self._t += 1
        b1t = self.beta1 ** self._t    # β₁ᵗ  (precomputed once per step)
        b2t = self.beta2 ** self._t    # β₂ᵗ

        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data

            # moment updates
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * grad * grad

            # bias correction
            m_hat = self._m[i] / (1.0 - b1t)
            v_hat = self._v[i] / (1.0 - b2t)

            # parameter update
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        self._zero_all()
        # NOTE: moment vectors (_m, _v) and step counter (_t) are NOT reset.
        # They are optimiser state, not gradient accumulators.
        # To reset Adam fully (e.g. for a fresh training run), construct
        # a new Adam instance.


# ═══════════════════════════════════════════════════════════
# VALIDATION HARNESS
# ═══════════════════════════════════════════════════════════

def _validate_all_optimizers():
    """
    Quick smoke test.  Run with:  python -c "from optimizers import _validate_all_optimizers; _validate_all_optimizers()"

    Trains XOR with each optimizer and prints the convergence table:
        Optimizer | LR | Epochs to < 0.01 | Final loss
    """
    # ── inline minimal MLP so this file is self-contained ───────────────
    import random
    from engine.value import Value   # adjust import path as needed

    random.seed(42)

    class Neuron:
        def __init__(self, nin):
            self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
            self.b = Value(0.0)
        def __call__(self, x):
            act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
            return act.tanh()
        def parameters(self):
            return self.w + [self.b]

    class Layer:
        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]
        def __call__(self, x):
            return [n(x) for n in self.neurons]
        def parameters(self):
            return [p for n in self.neurons for p in n.parameters()]

    class MLP:
        def __init__(self):
            self.l1 = Layer(2, 4)
            self.l2 = Layer(4, 4)
            self.l3 = Layer(4, 1)
        def __call__(self, x):
            x = self.l1(x)
            x = self.l2(x)
            return self.l3(x)[0]
        def parameters(self):
            return self.l1.parameters() + self.l2.parameters() + self.l3.parameters()

    XOR = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    def train(opt_class, opt_kwargs, max_epochs=2000):
        random.seed(42)
        model = MLP()
        opt   = opt_class(model.parameters(), **opt_kwargs)
        converged_at = None

        for epoch in range(1, max_epochs + 1):
            total_loss = Value(0.0)
            for xs, yt in XOR:
                pred = model([Value(v) for v in xs])
                diff = pred - Value(yt)
                total_loss = total_loss + diff * diff
            total_loss = total_loss * Value(0.25)  # mean

            opt.zero_grad()
            total_loss.backward()

            # scheduler hook for SGDStepDecay
            if hasattr(opt, 'scheduler_step'):
                opt.scheduler_step(epoch)

            opt.step()

            if converged_at is None and total_loss.data < 0.01:
                converged_at = epoch

        return converged_at, total_loss.data

    configs = [
        ("SGD fixed",            SGD,          {"lr": 2.807}),
        ("SGD step decay",       SGDStepDecay, {"lr": 2.807, "step_size": 100, "factor": 0.5}),
        ("SGD momentum β=0.9",   SGDMomentum,  {"lr": 0.1,   "beta": 0.9}),
        ("Adam α=0.001",         Adam,         {"lr": 0.001}),
    ]

    print(f"\n{'Optimizer':<25} {'LR':>8}  {'Epochs to <0.01':>17}  {'Final loss':>12}")
    print("─" * 68)
    for name, cls, kwargs in configs:
        ep, loss = train(cls, kwargs)
        ep_str   = str(ep) if ep else "did not converge"
        lr_val   = kwargs["lr"]
        print(f"{name:<25} {lr_val:>8.4f}  {ep_str:>17}  {loss:>12.6f}")
    print()


if __name__ == "__main__":
    _validate_all_optimizers()