"""
engine/optimizers.py
--------------------
All optimizers and LR schedulers for minigrad.

Design contract
---------------
Every optimizer subclasses Optimizer and implements:
    step()      — apply one gradient update to all parameters
    zero_grad() — reset all parameter .grad to 0.0

Every scheduler subclasses _LRScheduler and implements:
    get_lr()    — return list of new LRs, one per param_group
    step()      — advance one epoch/step and apply get_lr()

No optimizer may import anything outside the standard library.
No optimizer may touch any attribute of a Value except .data and .grad.

Optimizer inventory
-------------------
    SGD       — vanilla SGD; optional momentum, Nesterov, dampening
    AdaGrad   — adaptive per-parameter LR via accumulated squared grads
    RMSProp   — AdaGrad with exponential forgetting; optional centering
    Adam      — first + second moment estimates with bias correction
    AdamW     — Adam with decoupled weight decay (Loshchilov & Hutter 2019)

Scheduler inventory
-------------------
    StepLR            — multiply LR by gamma every step_size steps
    CosineAnnealingLR — cosine curve from base_lr to eta_min over T_max steps
    LinearWarmupLR    — linear ramp-up, then delegate to another scheduler

Factory
-------
    get_optimizer(name, parameters, **kwargs) — construct by string name

Gradient clipping
-----------------
    clip_grad_norm_(parameters, max_norm, norm_type=2.0)
    Call between loss.backward() and optimizer.step().
    Or attach permanently: optimizer.set_grad_clip(max_norm).

LR guidance
-----------
    SGD optimal for XOR:   ~2.807
    Adam canonical default:  0.001   <- NOT interchangeable with SGD's LR
    Using SGD's LR with Adam diverges immediately.

Step-decay guidance (XOR convergence zone [0.30, 2.81])
--------------------------------------------------------
    base_lr=2.807, gamma=0.1 -> LR after one step = 0.28  (below floor!)
    base_lr=2.807, gamma=0.5 -> 4 decays before LR < 0.30
    With step_size=100: first decay at epoch 100 -> safe for ~400 epochs.
    Rule: step_size >= 2 x epochs-to-first-meaningful-descent.
"""

import math


# =========================================================================
# GRADIENT CLIPPING  (module-level; usable standalone or via optimizer)
# =========================================================================

def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradient norms in-place.  Returns the global norm before clipping.

    Args:
        parameters  iterable of Value objects
        max_norm    clip threshold (must be > 0)
        norm_type   p-norm order; use math.inf for max-norm

    Returns:
        global_norm -- the pre-clip gradient norm (float)

    Typical use:
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    params = list(parameters)
    if not params:
        return 0.0

    if norm_type == math.inf:
        global_norm = max(abs(p.grad) for p in params)
    else:
        global_norm = sum(abs(p.grad) ** norm_type for p in params) ** (1.0 / norm_type)

    clip_coef = max_norm / (global_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad *= clip_coef

    return global_norm


# =========================================================================
# BASE OPTIMIZER
# =========================================================================

class Optimizer:
    """
    Abstract base class.

    Subclasses must override step().  zero_grad() is provided here via
    the shared _step_count and param_groups infrastructure.

    param_groups is a list of dicts; each dict has at minimum:
        "params"  -- list of Value objects
        "lr"      -- float learning rate
        plus any optimizer-specific hyperparameters

    This mirrors PyTorch's design so the mental model transfers.
    """

    def __init__(self, parameters, defaults: dict):
        params = list(parameters)
        if not params:
            raise ValueError(
                "Optimizer received an empty parameter list. "
                "Call model.parameters() before constructing the optimizer."
            )
        if "lr" in defaults and defaults["lr"] <= 0:
            raise ValueError(f"lr must be positive, got {defaults['lr']}")
        if "weight_decay" in defaults and defaults["weight_decay"] < 0:
            raise ValueError(f"weight_decay must be non-negative, got {defaults['weight_decay']}")

        self.param_groups    = []
        self._step_count     = 0
        self._clip_norm      = None
        self._clip_norm_type = 2.0

        self.add_param_group({"params": params, **defaults})

    def add_param_group(self, param_group: dict):
        """Add a new group of parameters with its own hyperparameters."""
        if "params" not in param_group:
            raise ValueError("param_group must contain a 'params' key")
        param_group = dict(param_group)
        param_group["params"] = list(param_group["params"])
        self.param_groups.append(param_group)
        self._init_group(param_group)

    def _init_group(self, group: dict):
        """Override to initialise per-group state (velocities, moments, etc.)."""
        pass

    def set_grad_clip(self, max_norm: float, norm_type: float = 2.0):
        """
        Attach gradient clipping to this optimizer.
        clip_grad_norm_ is called automatically at the start of step().
        Returns self for chaining: opt = SGD(...).set_grad_clip(1.0)
        """
        self._clip_norm      = max_norm
        self._clip_norm_type = norm_type
        return self

    def zero_grad(self):
        """Reset .grad to 0.0 on every parameter in all groups."""
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = 0.0

    def step(self):
        raise NotImplementedError(f"{type(self).__name__} must implement step()")

    def _maybe_clip(self):
        """Apply gradient clipping if configured via set_grad_clip()."""
        if self._clip_norm is not None:
            all_params = [p for g in self.param_groups for p in g["params"]]
            return clip_grad_norm_(all_params, self._clip_norm, self._clip_norm_type)
        return None

    # -- LR convenience property (operates on first param group) -----------

    @property
    def lr(self):
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value):
        self.param_groups[0]["lr"] = value

    # -- Serialisation ------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "step_count":   self._step_count,
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"}
                for g in self.param_groups
            ],
        }

    def load_state_dict(self, d: dict):
        self._step_count = d["step_count"]
        for g, saved in zip(self.param_groups, d["param_groups"]):
            g.update(saved)


# =========================================================================
# SGD  (vanilla + momentum + Nesterov + dampening)
# =========================================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov correction.

    Update rules
    ------------
    Vanilla SGD:
        theta <- theta - lr * g

    With momentum (beta > 0):
        v <- beta * v + (1 - dampening) * g
        theta <- theta - lr * v

    With Nesterov (dampening must be 0):
        v <- beta * v + g
        theta <- theta - lr * (beta * v + g)

    Nesterov g_effective = beta*v + g where v already contains g.
    This is PyTorch's formulation: buf.mul_(mom).add_(d_p) then
    d_p = d_p.add(buf, alpha=momentum).

    BUG FIXED: original had `g = beta * v + g` where v = beta*v_prev + (1-d)*g.
    This is correct. The previous code was also correct structurally but
    the comment was misleading. The implementation here matches PyTorch exactly.

    Args:
        parameters   iterable of Value objects
        lr           learning rate (must be > 0)
        momentum     beta in [0, 1)  (default 0.0 = vanilla SGD)
        nesterov     Nesterov correction (requires momentum > 0, dampening == 0)
        weight_decay L2 regularisation: g <- g + wd*theta  (default 0.0)
        dampening    dampens gradient accumulation into v (default 0.0)
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0, dampening=0.0):
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov requires momentum > 0")
        if nesterov and dampening != 0.0:
            raise ValueError("Nesterov is incompatible with dampening != 0")

        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=weight_decay, dampening=dampening
        )
        super().__init__(parameters, defaults)

    def _init_group(self, group: dict):
        group["velocities"] = [0.0] * len(group["params"])

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr         = group["lr"]
            beta       = group["momentum"]
            nesterov   = group["nesterov"]
            wd         = group["weight_decay"]
            dampening  = group["dampening"]
            velocities = group["velocities"]

            for i, p in enumerate(group["params"]):
                g = p.grad
                if wd != 0.0:
                    g = g + wd * p.data

                if beta != 0.0:
                    # v <- beta*v + (1 - dampening)*g
                    v = beta * velocities[i] + (1.0 - dampening) * g
                    velocities[i] = v
                    # Nesterov: g_eff = beta*v + g  (v already includes g)
                    # Standard: g_eff = v
                    g = beta * v + g if nesterov else v

                p.data -= lr * g

        self._step_count += 1

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["velocities"] = [g["velocities"] for g in self.param_groups]
        return d

    def load_state_dict(self, d: dict):
        super().load_state_dict(d)
        for g, vels in zip(self.param_groups, d.get("velocities", [])):
            g["velocities"] = vels

    def __repr__(self):
        g = self.param_groups[0]
        n = sum(len(g["params"]) for g in self.param_groups)
        return (f"SGD(lr={g['lr']}, momentum={g['momentum']}, "
                f"nesterov={g['nesterov']}, params={n})")


# =========================================================================
# ADAGRAD
# =========================================================================

class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm  (Duchi et al., 2011)

    Accumulates squared gradients and normalises the LR per-parameter:
        G_i <- G_i + g_i^2
        theta_i <- theta_i - (clr / sqrt(G_i + eps)) * g_i

    clr optionally decays over time:
        clr = lr / (1 + step * lr_decay)

    Strength: sparse gradients -- rare features get a larger effective LR.
    Weakness: G accumulates forever -> LR -> 0 on long runs (fixed by RMSProp).

    BUG FIXED: original applied weight_decay twice -- once as `g = p.grad + wd*p.data`
    and again inside the loop body.  Now applied exactly once.

    Args:
        parameters   iterable of Value objects
        lr           initial learning rate (default 0.01)
        eps          numerical floor (default 1e-10)
        weight_decay L2 regularisation (default 0.0)
        lr_decay     global LR shrinkage per step (default 0.0 = off)
    """

    def __init__(self, parameters, lr=0.01, eps=1e-10,
                 weight_decay=0.0, lr_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, lr_decay=lr_decay)
        super().__init__(parameters, defaults)

    def _init_group(self, group: dict):
        group["sum_sq"] = [0.0] * len(group["params"])

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr       = group["lr"]
            eps      = group["eps"]
            wd       = group["weight_decay"]
            lr_decay = group["lr_decay"]
            sum_sq   = group["sum_sq"]

            # FIX: compute clr once outside the param loop
            clr = lr / (1.0 + self._step_count * lr_decay)

            for i, p in enumerate(group["params"]):
                # FIX: weight_decay applied exactly once
                g = p.grad + (wd * p.data if wd != 0.0 else 0.0)
                sum_sq[i] += g * g
                p.data -= clr * g / (math.sqrt(sum_sq[i]) + eps)

        self._step_count += 1

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["sum_sq"] = [g["sum_sq"] for g in self.param_groups]
        return d

    def load_state_dict(self, d: dict):
        super().load_state_dict(d)
        for g, sq in zip(self.param_groups, d.get("sum_sq", [])):
            g["sum_sq"] = sq

    def __repr__(self):
        g = self.param_groups[0]
        n = sum(len(g["params"]) for g in self.param_groups)
        return f"AdaGrad(lr={g['lr']}, eps={g['eps']}, params={n})"


# =========================================================================
# RMSPROP
# =========================================================================

class RMSProp(Optimizer):
    """
    RMSProp  (Hinton, unpublished 2012)

    Fixes AdaGrad's vanishing LR with an exponential moving average:
        E[g^2]_t <- alpha * E[g^2]_{t-1} + (1 - alpha) * g^2
        theta <- theta - lr * g / (sqrt(E[g^2]) + eps)

    Centered variant (centered=True):
        variance = E[g^2] - E[g]^2  (true variance, more stable)
        denom = sqrt(max(variance, 0)) + eps  <- clamped for float safety

    With momentum:
        buf <- mom * buf + g / denom
        theta <- theta - lr * buf

    Args:
        parameters   iterable of Value objects
        lr           learning rate (default 0.01)
        alpha        smoothing factor in [0,1) (default 0.99)
        eps          numerical floor (default 1e-8)
        momentum     momentum factor in [0,1) (default 0.0)
        weight_decay L2 regularisation (default 0.0)
        centered     use variance instead of raw 2nd moment (default False)
    """

    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8,
                 momentum=0.0, weight_decay=0.0, centered=False):
        defaults = dict(
            lr=lr, alpha=alpha, eps=eps,
            momentum=momentum, weight_decay=weight_decay, centered=centered
        )
        super().__init__(parameters, defaults)

    def _init_group(self, group: dict):
        n = len(group["params"])
        group["sq_avg"]       = [0.0] * n
        group["momentum_buf"] = [0.0] * n
        if group["centered"]:
            group["grad_avg"] = [0.0] * n

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr       = group["lr"]
            alpha    = group["alpha"]
            eps      = group["eps"]
            mom      = group["momentum"]
            wd       = group["weight_decay"]
            centered = group["centered"]
            sq_avg   = group["sq_avg"]
            mbuf     = group["momentum_buf"]
            grad_avg = group.get("grad_avg")

            for i, p in enumerate(group["params"]):
                g = p.grad + (wd * p.data if wd != 0.0 else 0.0)

                sq_avg[i] = alpha * sq_avg[i] + (1.0 - alpha) * g * g

                if centered:
                    grad_avg[i] = alpha * grad_avg[i] + (1.0 - alpha) * g
                    # FIX: clamp variance to >= 0 to guard against float error
                    variance = sq_avg[i] - grad_avg[i] ** 2
                    denom = math.sqrt(max(variance, 0.0)) + eps
                else:
                    denom = math.sqrt(sq_avg[i]) + eps

                if mom != 0.0:
                    mbuf[i] = mom * mbuf[i] + g / denom
                    p.data -= lr * mbuf[i]
                else:
                    p.data -= lr * g / denom

        self._step_count += 1

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["sq_avg"]       = [g["sq_avg"] for g in self.param_groups]
        d["momentum_buf"] = [g["momentum_buf"] for g in self.param_groups]
        return d

    def load_state_dict(self, d: dict):
        super().load_state_dict(d)
        for g, sq, mb in zip(
            self.param_groups,
            d.get("sq_avg", []),
            d.get("momentum_buf", []),
        ):
            g["sq_avg"]       = sq
            g["momentum_buf"] = mb

    def __repr__(self):
        g = self.param_groups[0]
        n = sum(len(g["params"]) for g in self.param_groups)
        return f"RMSProp(lr={g['lr']}, alpha={g['alpha']}, params={n})"


# =========================================================================
# ADAM
# =========================================================================

class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation  (Kingma & Ba, 2015)

    Per-parameter first (mean) and second (variance) moment estimates
    with bias correction for the early steps where m, v are near zero:

        m  <- beta1*m + (1-beta1)*g          first moment
        v  <- beta2*v + (1-beta2)*g^2        second moment
        m_hat = m / (1 - beta1^t)            bias-corrected
        v_hat = v / (1 - beta2^t)
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

    AMSGrad variant (amsgrad=True):
        Replaces v_hat with max(v_hat_prev, v_hat), guaranteeing a
        non-increasing effective LR, which aids convergence on
        non-stationary objectives.

    BUG FIXED (vs original): per-parameter state was keyed by id(p),
    which is not stable across restarts or after pickling.  State is now
    stored by positional index within the param group, matching PyTorch's
    checkpoint format.

    CRITICAL LR NOTE:
        SGD optimal for XOR:  ~2.807
        Adam canonical:        0.001
        They are NOT interchangeable.  Using SGD's LR with Adam diverges.

    Args:
        parameters   iterable of Value objects
        lr           step size alpha (default 1e-3)
        betas        (beta1, beta2) moment decay rates (default (0.9, 0.999))
        eps          numerical floor (default 1e-8)
        weight_decay L2 regularisation: g <- g + wd*theta (default 0.0)
        amsgrad      use AMSGrad variant (default False)
    """

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, amsgrad=False):
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        super().__init__(parameters, defaults)

    def _init_group(self, group: dict):
        n = len(group["params"])
        group["m"]     = [0.0] * n    # first moments
        group["v"]     = [0.0] * n    # second moments
        group["v_max"] = [0.0] * n    # AMSGrad running max
        group["step"]  = [0]   * n    # per-parameter step counter

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr             = group["lr"]
            beta1, beta2   = group["betas"]
            eps            = group["eps"]
            wd             = group["weight_decay"]
            amsgrad        = group["amsgrad"]
            m_list         = group["m"]
            v_list         = group["v"]
            vmax           = group["v_max"]
            steps          = group["step"]

            for i, p in enumerate(group["params"]):
                g = p.grad + (wd * p.data if wd != 0.0 else 0.0)

                steps[i] += 1
                t = steps[i]

                m_list[i] = beta1 * m_list[i] + (1.0 - beta1) * g
                v_list[i] = beta2 * v_list[i] + (1.0 - beta2) * g * g

                m_hat = m_list[i] / (1.0 - beta1 ** t)
                v_hat = v_list[i] / (1.0 - beta2 ** t)

                if amsgrad:
                    vmax[i] = max(vmax[i], v_hat)
                    denom = math.sqrt(vmax[i]) + eps
                else:
                    denom = math.sqrt(v_hat) + eps

                p.data -= lr * m_hat / denom

        self._step_count += 1

    def state_dict(self) -> dict:
        d = super().state_dict()
        # positional index keys -- stable across restarts
        d["m"]     = [g["m"]     for g in self.param_groups]
        d["v"]     = [g["v"]     for g in self.param_groups]
        d["v_max"] = [g["v_max"] for g in self.param_groups]
        d["step"]  = [g["step"]  for g in self.param_groups]
        return d

    def load_state_dict(self, d: dict):
        super().load_state_dict(d)
        for g, m, v, vm, st in zip(
            self.param_groups,
            d.get("m",     []),
            d.get("v",     []),
            d.get("v_max", []),
            d.get("step",  []),
        ):
            g["m"]     = m
            g["v"]     = v
            g["v_max"] = vm
            g["step"]  = st

    def __repr__(self):
        g = self.param_groups[0]
        n = sum(len(g["params"]) for g in self.param_groups)
        return (f"Adam(lr={g['lr']}, betas={g['betas']}, "
                f"amsgrad={g['amsgrad']}, params={n})")


# =========================================================================
# ADAMW  (decoupled weight decay)
# =========================================================================

class AdamW(Adam):
    """
    AdamW: Adam with Decoupled Weight Decay  (Loshchilov & Hutter, 2019)

    Standard Adam folds weight decay into the gradient before the moment
    update, which couples regularisation to the adaptive scaling.  AdamW
    applies weight decay directly to the parameters, independently:

        [same Adam moment updates on raw gradient g -- NO wd in g]
        theta <- theta * (1 - lr * wd)      decoupled weight decay
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

    This is the preferred optimizer for transformers / language models.

    Args:
        parameters   iterable of Value objects
        lr           step size (default 1e-3)
        betas        (beta1, beta2) (default (0.9, 0.999))
        eps          (default 1e-8)
        weight_decay decoupled regularisation coefficient (default 0.01)
        amsgrad      AMSGrad variant (default False)
    """

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01, amsgrad=False):
        # Pass weight_decay=0 to Adam so it never adds wd to the gradient;
        # we apply decoupled_wd ourselves in step().
        super().__init__(
            parameters, lr=lr, betas=betas, eps=eps,
            weight_decay=0.0, amsgrad=amsgrad
        )
        for group in self.param_groups:
            group["decoupled_wd"] = weight_decay

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["decoupled_wd"]
            amsgrad      = group["amsgrad"]
            m_list       = group["m"]
            v_list       = group["v"]
            vmax         = group["v_max"]
            steps        = group["step"]

            for i, p in enumerate(group["params"]):
                g = p.grad   # raw gradient -- weight decay NOT added here

                steps[i] += 1
                t = steps[i]

                m_list[i] = beta1 * m_list[i] + (1.0 - beta1) * g
                v_list[i] = beta2 * v_list[i] + (1.0 - beta2) * g * g

                m_hat = m_list[i] / (1.0 - beta1 ** t)
                v_hat = v_list[i] / (1.0 - beta2 ** t)

                if amsgrad:
                    vmax[i] = max(vmax[i], v_hat)
                    denom = math.sqrt(vmax[i]) + eps
                else:
                    denom = math.sqrt(v_hat) + eps

                # Decoupled: apply wd directly to parameter, then Adam step
                p.data *= (1.0 - lr * wd)
                p.data -= lr * m_hat / denom

        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        n = sum(len(g["params"]) for g in self.param_groups)
        return (f"AdamW(lr={g['lr']}, betas={g['betas']}, "
                f"wd={g['decoupled_wd']}, params={n})")


# =========================================================================
# LR SCHEDULERS
# =========================================================================

class _LRScheduler:
    """
    Abstract base class for LR schedulers.

    Usage pattern:
        optimizer = SGD(model.parameters(), lr=2.807)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        for epoch in range(epochs):
            train_one_epoch(...)
            scheduler.step()     # call AFTER optimizer.step()

    last_step starts at -1; __init__ calls self.step() once to advance to
    step 0 and apply the initial LR.  This matches PyTorch's convention.
    """

    def __init__(self, optimizer, last_step: int = -1):
        self.optimizer = optimizer
        self.last_step = last_step
        self.base_lrs  = [g["lr"] for g in optimizer.param_groups]
        self.step()    # advance to 0, apply initial LR

    def get_lr(self) -> list:
        raise NotImplementedError(f"{type(self).__name__} must implement get_lr()")

    def step(self):
        self.last_step += 1
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def get_last_lr(self) -> list:
        """Return the most recently applied LR for each param group."""
        return [g["lr"] for g in self.optimizer.param_groups]


class StepLR(_LRScheduler):
    """
    Multiply LR by gamma every step_size steps.

    Formula:  lr_t = base_lr * gamma ^ (t // step_size)

    Decay schedule for base_lr=2.807, gamma=0.5, step_size=100:
        epoch   0-99:   2.807  (safe, in XOR convergence zone)
        epoch 100-199:  1.404  (safe)
        epoch 200-299:  0.702  (safe)
        epoch 300-399:  0.351  (barely safe, just above 0.30 floor)
        epoch 400+:     0.176  (below floor -- network stalls)

    WARNING: gamma=0.1 (the default) is almost always too aggressive for
    small networks.  Use gamma=0.5 for XOR.  For larger networks use
    CosineAnnealingLR instead to avoid the abrupt drop.

    Args:
        optimizer   Optimizer instance
        step_size   decay interval (epochs/steps), must be >= 1
        gamma       multiplicative factor, 0 < gamma < 1 (default 0.1)
        last_step   resume step (-1 = fresh start)
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1,
                 last_step: int = -1):
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.step_size = step_size
        self.gamma     = gamma
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list:
        decay = self.gamma ** (self.last_step // self.step_size)
        return [base * decay for base in self.base_lrs]

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"


class CosineAnnealingLR(_LRScheduler):
    """
    Smooth cosine decay from base_lr to eta_min over T_max steps.

        lr_t = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(pi * t / T_max))

    At t=0:     lr = base_lr
    At t=T_max: lr = eta_min

    Advantage over StepLR: no sudden drops -- the network never experiences
    a discontinuous change in step size.  Preferred for any network where
    you know the total training budget ahead of time.

    Args:
        optimizer  Optimizer instance
        T_max      number of steps for one cosine half-period
        eta_min    minimum LR floor (default 0.0)
        last_step  resume step (-1 = fresh start)
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0,
                 last_step: int = -1):
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        self.T_max   = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list:
        t = min(self.last_step, self.T_max)   # clamp -- no extrapolation
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))
        return [
            self.eta_min + (base - self.eta_min) * cos_factor
            for base in self.base_lrs
        ]

    def __repr__(self):
        return f"CosineAnnealingLR(T_max={self.T_max}, eta_min={self.eta_min})"


class LinearWarmupLR(_LRScheduler):
    """
    Linear warmup for warmup_steps steps, then delegate to after_scheduler.

    Ramp phase (step < warmup_steps):
        lr = base_lr * (step + 1) / warmup_steps

    After warmup (step >= warmup_steps):
        lr = after_scheduler.get_lr() evaluated at (step - warmup_steps)

    BUG FIXED: the original mutated after_scheduler.last_step permanently,
    which corrupted the after_scheduler's internal state if it was also
    referenced elsewhere.  This version saves and restores last_step around
    the get_lr() call, leaving the after_scheduler's state untouched.

    Args:
        optimizer        Optimizer instance
        warmup_steps     number of ramp-up steps (must be >= 1)
        after_scheduler  _LRScheduler instance to use post-warmup (or None)
        last_step        resume step (-1 = fresh start)
    """

    def __init__(self, optimizer, warmup_steps: int,
                 after_scheduler=None, last_step: int = -1):
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        self.warmup_steps    = warmup_steps
        self.after_scheduler = after_scheduler
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list:
        t = self.last_step

        if t < self.warmup_steps:
            scale = (t + 1) / self.warmup_steps
            return [base * scale for base in self.base_lrs]

        if self.after_scheduler is not None:
            # FIX: save and restore last_step -- do not mutate permanently
            saved = self.after_scheduler.last_step
            self.after_scheduler.last_step = t - self.warmup_steps
            lrs = self.after_scheduler.get_lr()
            self.after_scheduler.last_step = saved
            return lrs

        return list(self.base_lrs)   # hold at base_lr if no after_scheduler

    def __repr__(self):
        return (f"LinearWarmupLR(warmup_steps={self.warmup_steps}, "
                f"after={self.after_scheduler!r})")


# =========================================================================
# FACTORY
# =========================================================================

_REGISTRY = {
    "sgd":      SGD,
    "adagrad":  AdaGrad,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "adamw":    AdamW,
}


def get_optimizer(name: str, parameters, **kwargs):
    """
    Construct an optimizer by string name.

    Examples:
        opt = get_optimizer("adam", model.parameters(), lr=0.001)
        opt = get_optimizer("sgd",  model.parameters(), lr=2.807, momentum=0.9)

    Args:
        name        case-insensitive: 'sgd', 'adagrad', 'rmsprop', 'adam', 'adamw'
        parameters  iterable of Value objects
        **kwargs    forwarded verbatim to the optimizer constructor

    Raises:
        ValueError if name is not in the registry.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](parameters, **kwargs)

#just for the test cases, not for general use