"""
optimizers.py
=============
Production-grade optimizer suite built on the Value autograd engine.
Every optimizer is an external agent: reads .grad, writes .data, has
no knowledge of graph structure.

Optimizers implemented
----------------------
  SGD            — vanilla, momentum, Nesterov momentum
  AdaGrad        — per-parameter adaptive learning rate (accumulating)
  RMSProp        — per-parameter adaptive learning rate (exponential decay)
  Adam           — adaptive moments with bias correction (Kingma & Ba 2015)
  AdamW          — Adam with decoupled weight decay (Loshchilov & Hutter 2019)

Learning rate schedulers
------------------------
  StepLR         — multiply lr by γ every `step_size` optimizer steps
  CosineAnnealingLR — cosine decay from lr_max to lr_min over T_max steps
  LinearWarmupLR — linear ramp from 0 to base_lr over warmup_steps,
                   then hand off to a wrapped scheduler

Regularization / stability
--------------------------
  Weight decay   — L2 penalty; coupled (SGD-style) or decoupled (AdamW-style)
  Gradient clip  — global ℓ₂ norm clipping applied before every step;
                   available as a standalone function and as an optimizer mixin

Derivations
-----------

SGD (vanilla)
    p_t = p_{t-1} - lr · g_t

SGD + Momentum (dampened heavy-ball)
    v_t = β · v_{t-1} + g_t              # velocity accumulates gradient
    p_t = p_{t-1} - lr · v_t

    Intuition: v_t is an exponential moving average of gradients scaled by
    1/(1-β). In directions where gradients are consistent, v builds up and
    we take larger steps. In oscillating directions, updates cancel and we
    damp.

SGD + Nesterov Momentum
    v_t  = β · v_{t-1} + g_t
    p_t  = p_{t-1} - lr · (β · v_t + g_t)

    Equivalent to evaluating the gradient at the "look-ahead" position
    p - lr·β·v. Converges faster on convex problems; gradient at the future
    point is a better signal than the gradient at the current point.

    Derivation sketch:
        Look-ahead: p̃ = p - lr·β·v_{t-1}
        Gradient there ≈ g_t  (same as standard for a single step)
        Update: p = p̃ - lr·g_t
               = p - lr·β·v_{t-1} - lr·g_t
               = p - lr·(β·v_{t-1} + g_t)
               = p - lr·(v_t - β·v_{t-1} + β·v_{t-1})
          Rewritten in terms of v_t: p = p - lr·(β·v_t + g_t)  ✓

AdaGrad (Duchi et al. 2011)
    G_t  = G_{t-1} + g_t²              # accumulated squared gradients
    p_t  = p_{t-1} - lr · g_t / (√G_t + ε)

    Parameters that have seen large historical gradients get a smaller
    effective lr. Excellent for sparse features; problematic for non-convex
    deep nets because G_t only grows → lr → 0.

RMSProp (Hinton, unpublished 2012)
    v_t  = β · v_{t-1} + (1 - β) · g_t²   # EMA of squared gradients
    p_t  = p_{t-1} - lr · g_t / (√v_t + ε)

    Fixes AdaGrad's vanishing lr by replacing cumulative sum with EMA.
    β = 0.9 is the standard; v_t ≈ (1-β)·Σ β^k · g_{t-k}².

Adam (Kingma & Ba 2015)
    m_t  = β1 · m_{t-1} + (1 - β1) · g_t      # 1st moment (mean)
    v_t  = β2 · v_{t-1} + (1 - β2) · g_t²     # 2nd moment (uncentered variance)

    Bias correction (both moments are 0-initialised, so early estimates
    are biased toward 0):
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)

    Update:
        p_t  = p_{t-1} - lr · m̂_t / (√v̂_t + ε)

    Effective step size ≈ lr · (1 - β2^t)^½ / (1 - β1^t), which starts
    small (bias-correction warming up) and converges to lr.

AdamW (Loshchilov & Hutter 2019) — decoupled weight decay
    Adam's L2 regularisation (adding λ·p to the gradient) is NOT equivalent
    to weight decay when the update is scaled by the adaptive step.
    AdamW decouples decay from the gradient statistics:

        p_t  = (1 - lr·λ) · p_{t-1} - lr · m̂_t / (√v̂_t + ε)

    Concretely: the decay shrinks p proportionally regardless of the local
    curvature estimate. This is the correct L2 regularisation for adaptive
    optimizers.

Gradient Clipping (global ℓ₂ norm)
    global_norm = √(Σ_i g_i²)
    if global_norm > clip_norm:
        g_i ← g_i · clip_norm / global_norm   for all i

    Keeps the gradient vector's direction, scales its magnitude down to
    clip_norm. Essential for RNNs; improves stability for deep nets generally.

Weight Decay (L2 regularisation, coupled — SGD-style)
    L_reg = L + (λ/2) · Σ p²
    ∂L_reg/∂p = g + λ · p
    p_t = p_{t-1} - lr · (g_t + λ · p_{t-1})

    "Coupled" because the decay is folded into the gradient before the
    optimizer update formula. For SGD this is identical to AdamW-style
    decoupled decay. For Adam it is NOT — see AdamW above.

StepLR
    lr_t = lr_0 · γ^⌊t / step_size⌋

    Every `step_size` optimizer steps, multiply current lr by γ.
    Simple but requires hand-tuning of step_size and γ.

CosineAnnealingLR (Loshchilov & Hutter 2017, SGDR)
    lr_t = lr_min + ½ · (lr_max - lr_min) · (1 + cos(π · t / T_max))

    Smoothly decays from lr_max → lr_min over T_max steps following a
    cosine curve. Reaches lr_min at t = T_max, then can be restarted.
    The half-cosine shape keeps lr near lr_max longer, then drops sharply —
    empirically better than linear decay for SGD.

LinearWarmupLR
    if t < warmup_steps:
        lr_t = lr_base · t / warmup_steps     # linear ramp from 0
    else:
        lr_t = wrapped_scheduler.get_lr()      # hand off after warmup

    Warmup prevents large early updates when moments/accumulators are
    uninitialized (especially important for Adam where early m̂, v̂ are noisy).
"""

import math


# ═══════════════════════════════════════════════════════════════════════════ #
#  Utility                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip the global ℓ_p norm of gradients across all parameters in-place.

    Computes the global norm once, then rescales every gradient uniformly
    so the global norm equals max_norm. Parameters whose gradient is already
    below max_norm are untouched.

    Parameters
    ----------
    parameters : iterable of Value
    max_norm   : float — maximum allowed global gradient norm
    norm_type  : float — order of the norm (default 2 = Euclidean)

    Returns
    -------
    float — global norm *before* clipping (useful for logging)
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    params = list(parameters)
    if norm_type == math.inf:
        global_norm = max(abs(p.grad) for p in params) if params else 0.0
    else:
        global_norm = sum(abs(p.grad) ** norm_type for p in params) ** (1.0 / norm_type)

    clip_coef = max_norm / (global_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad *= clip_coef

    return global_norm


# ═══════════════════════════════════════════════════════════════════════════ #
#  Base Optimizer                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #


class Optimizer:
    """
    Abstract base for all optimizers.

    Subclasses must implement `step()`.
    Provides: zero_grad(), add_param_group(), grad_clip support.
    """

    def __init__(self, parameters, defaults: dict):
        self.param_groups = []
        self._step_count = 0          # total step() calls across lifetime
        self._clip_norm = None        # set via set_grad_clip()
        self._clip_norm_type = 2.0

        params = list(parameters)
        if not params:
            raise ValueError("optimizer received an empty parameter list")

        # Validate defaults
        if "lr" in defaults and defaults["lr"] <= 0:
            raise ValueError(f"Learning rate must be positive, got {defaults['lr']}")
        if "weight_decay" in defaults and defaults["weight_decay"] < 0:
            raise ValueError(f"weight_decay must be non-negative, got {defaults['weight_decay']}")

        self.add_param_group({"params": params, **defaults})

    def add_param_group(self, param_group: dict):
        """
        Add a new parameter group (e.g. different lr for different layers).
        `param_group` must contain a 'params' key.
        """
        if "params" not in param_group:
            raise ValueError("param_group must contain a 'params' key")
        param_group = dict(param_group)
        param_group["params"] = list(param_group["params"])
        self.param_groups.append(param_group)
        # Initialise per-parameter state for this group
        self._init_group(param_group)

    def _init_group(self, group: dict):
        """Override in subclasses to set up per-parameter state."""
        pass

    def set_grad_clip(self, max_norm: float, norm_type: float = 2.0):
        """
        Enable automatic gradient clipping before every step().
        Set max_norm=None to disable.
        """
        self._clip_norm = max_norm
        self._clip_norm_type = norm_type
        return self

    def zero_grad(self):
        """
        Zero all parameter gradients.
        Call *before* each forward pass, not after step(), so the graph
        is built against clean gradients.
        """
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = 0.0

    def step(self):
        raise NotImplementedError

    def _maybe_clip(self):
        """Apply gradient clipping if configured."""
        if self._clip_norm is not None:
            all_params = [p for g in self.param_groups for p in g["params"]]
            return clip_grad_norm_(all_params, self._clip_norm, self._clip_norm_type)
        return None

    @property
    def lr(self):
        """Convenience: return the lr of the first param group."""
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value):
        """Convenience: set the lr of the first param group."""
        self.param_groups[0]["lr"] = value

    def state_dict(self) -> dict:
        """Snapshot of optimizer state for checkpointing."""
        return {
            "step_count": self._step_count,
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"}
                for g in self.param_groups
            ],
            "state": getattr(self, "state", {}),
        }

    def load_state_dict(self, d: dict):
        """Restore from a snapshot."""
        self._step_count = d["step_count"]
        for g, saved in zip(self.param_groups, d["param_groups"]):
            g.update(saved)
        if hasattr(self, "state"):
            self.state.update(d.get("state", {}))


# ═══════════════════════════════════════════════════════════════════════════ #
#  SGD — vanilla · momentum · Nesterov                                       #
# ═══════════════════════════════════════════════════════════════════════════ #


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov
    look-ahead, weight decay, and gradient clipping.

    Parameters
    ----------
    parameters   : iterable of Value
    lr           : float   — learning rate (required, > 0)
    momentum     : float   — momentum coefficient β ∈ [0, 1)
                             0 = vanilla SGD
    nesterov     : bool    — use Nesterov look-ahead (requires momentum > 0)
    weight_decay : float   — L2 penalty coefficient λ ≥ 0 (coupled)
    dampening    : float   — dampening for momentum: v = β·v + (1-d)·g
                             0 = standard momentum; incompatible with nesterov

    Update rules
    ------------
    Vanilla:
        p -= lr · g

    Momentum:
        v = β·v + (1 - dampening)·g
        p -= lr · v

    Nesterov:
        v = β·v + g
        p -= lr · (β·v + g)

    Weight decay (coupled, applied before momentum):
        g ← g + λ·p
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if nesterov and (momentum == 0.0 or dampening != 0.0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            dampening=dampening,
        )
        super().__init__(parameters, defaults)

    def _init_group(self, group):
        """Allocate velocity buffer v=0 for each parameter."""
        group["velocities"] = [0.0] * len(group["params"])

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr           = group["lr"]
            beta         = group["momentum"]
            nesterov     = group["nesterov"]
            wd           = group["weight_decay"]
            dampening    = group["dampening"]
            velocities   = group["velocities"]

            for i, p in enumerate(group["params"]):
                g = p.grad

                # ── Weight decay (coupled L2) ───────────────────────────────
                if wd != 0.0:
                    g = g + wd * p.data

                # ── Momentum ───────────────────────────────────────────────
                if beta != 0.0:
                    v = velocities[i]
                    v = beta * v + (1.0 - dampening) * g
                    velocities[i] = v

                    if nesterov:
                        # Nesterov: use β·v + g as the effective gradient
                        # p -= lr · (β·v + g)  where v is already updated
                        g = beta * v + g
                    else:
                        g = v

                p.data -= lr * g

        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        return (
            f"SGD(lr={g['lr']}, momentum={g['momentum']}, "
            f"nesterov={g['nesterov']}, weight_decay={g['weight_decay']}, "
            f"params={sum(len(g['params']) for g in self.param_groups)})"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  AdaGrad                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #


class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm (Duchi et al. 2011).

    Per-parameter learning rate adapted by the cumulative sum of squared
    gradients. Ideal for sparse data; lr effectively vanishes for dense
    parameters over long training runs.

    Parameters
    ----------
    parameters    : iterable of Value
    lr            : float  — global learning rate η
    eps           : float  — numerical stability term ε (default 1e-10)
    weight_decay  : float  — L2 penalty (coupled)
    lr_decay      : float  — decay applied to lr itself each step:
                             lr_t = lr / (1 + t · lr_decay)

    Update rule
    -----------
        G_t += g_t²
        p_t  = p_{t-1} - (lr_t / (√G_t + ε)) · g_t
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        lr_decay: float = 0.0,
    ):
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, lr_decay=lr_decay)
        super().__init__(parameters, defaults)

    def _init_group(self, group):
        group["sum_sq"] = [0.0] * len(group["params"])  # G accumulator

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr           = group["lr"]
            eps          = group["eps"]
            wd           = group["weight_decay"]
            lr_decay     = group["lr_decay"]
            sum_sq       = group["sum_sq"]

            # Apply lr decay
            clr = lr / (1.0 + self._step_count * lr_decay)

            for i, p in enumerate(group["params"]):
                g = p.grad

                if wd != 0.0:
                    g = g + wd * p.data

                sum_sq[i] += g * g
                p.data -= clr * g / (math.sqrt(sum_sq[i]) + eps)

        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        return f"AdaGrad(lr={g['lr']}, eps={g['eps']}, params={sum(len(g['params']) for g in self.param_groups)})"


# ═══════════════════════════════════════════════════════════════════════════ #
#  RMSProp                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation (Hinton, Coursera Lecture 2012).

    Replaces AdaGrad's cumulative sum with an exponential moving average
    of squared gradients, preventing the learning rate from vanishing.

    Parameters
    ----------
    parameters    : iterable of Value
    lr            : float  — global learning rate
    alpha         : float  — smoothing (decay) coefficient β (default 0.99)
    eps           : float  — numerical stability ε
    momentum      : float  — optional momentum on top of RMSProp (default 0)
    weight_decay  : float  — L2 penalty (coupled)
    centered      : bool   — if True, normalise by E[g]² - E[g²] (variance)
                             instead of E[g²]; more stable but costlier

    Update rule (standard)
    ----------------------
        v_t = α · v_{t-1} + (1 - α) · g_t²
        p_t = p_{t-1} - lr · g_t / (√v_t + ε)

    With momentum:
        buf_t = momentum · buf_{t-1} + g_t / (√v_t + ε)
        p_t   = p_{t-1} - lr · buf_t

    Centered (variance normalisation):
        g_mean_t = α · g_mean_{t-1} + (1 - α) · g_t
        v_t      = α · v_{t-1} + (1 - α) · g_t²
        var_t    = v_t - g_mean_t²
        p_t      = p_{t-1} - lr · g_t / (√var_t + ε)
    """

    def __init__(
        self,
        parameters,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        centered: bool = False,
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        defaults = dict(
            lr=lr, alpha=alpha, eps=eps,
            momentum=momentum, weight_decay=weight_decay, centered=centered,
        )
        super().__init__(parameters, defaults)

    def _init_group(self, group):
        n = len(group["params"])
        group["sq_avg"]   = [0.0] * n   # v: EMA of g²
        group["momentum_buf"] = [0.0] * n
        if group["centered"]:
            group["grad_avg"] = [0.0] * n   # EMA of g (for centering)

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
            grad_avg = group.get("grad_avg", None)

            for i, p in enumerate(group["params"]):
                g = p.grad

                if wd != 0.0:
                    g = g + wd * p.data

                sq_avg[i] = alpha * sq_avg[i] + (1.0 - alpha) * g * g

                if centered:
                    grad_avg[i] = alpha * grad_avg[i] + (1.0 - alpha) * g
                    denom = math.sqrt(sq_avg[i] - grad_avg[i] ** 2) + eps
                else:
                    denom = math.sqrt(sq_avg[i]) + eps

                if mom != 0.0:
                    mbuf[i] = mom * mbuf[i] + g / denom
                    p.data -= lr * mbuf[i]
                else:
                    p.data -= lr * g / denom

        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        return (
            f"RMSProp(lr={g['lr']}, alpha={g['alpha']}, centered={g['centered']}, "
            f"params={sum(len(g['params']) for g in self.param_groups)})"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Adam                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Kingma & Ba, 2015).

    Combines momentum (first moment) with per-parameter adaptive lr (second
    moment). Bias-corrected estimates ensure the effective step size is
    well-calibrated from step 1.

    Parameters
    ----------
    parameters    : iterable of Value
    lr            : float        — step size α (default 1e-3)
    betas         : (float,float)— (β1, β2) moment decay rates
    eps           : float        — denominator stability ε
    weight_decay  : float        — L2 penalty, folded into gradient (coupled)
                                   For decoupled decay use AdamW instead.
    amsgrad       : bool         — use AMSGrad variant (Reddi et al. 2018):
                                   maintains the max of all past v̂ to
                                   guarantee non-increasing effective lr

    Update rule
    -----------
        g_t  = ∇L (+ λ·p if weight_decay)
        m_t  = β1·m_{t-1} + (1 - β1)·g_t
        v_t  = β2·v_{t-1} + (1 - β2)·g_t²

        Bias correction:
            m̂_t = m_t / (1 - β1^t)
            v̂_t = v_t / (1 - β2^t)

        AMSGrad:
            v̂_max_t = max(v̂_max_{t-1}, v̂_t)
            denom    = √v̂_max_t + ε

        Standard:
            denom = √v̂_t + ε

        p_t = p_{t-1} - lr · m̂_t / denom
    """

    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(parameters, defaults)
        # Per-parameter state (shared across all groups)
        self.state = {}   # id(param) → {"step", "m", "v", "v_max"}

    def _init_group(self, group):
        # State is lazily initialised on first step (we don't have param ids yet
        # at add_param_group time if parameters are constructed later)
        pass

    def _get_state(self, p, amsgrad: bool):
        pid = id(p)
        if pid not in self.state:
            self.state[pid] = {
                "step": 0,
                "m": 0.0,      # 1st moment
                "v": 0.0,      # 2nd moment
                "v_max": 0.0 if amsgrad else None,
            }
        return self.state[pid]

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["weight_decay"]
            amsgrad      = group["amsgrad"]

            for p in group["params"]:
                g = p.grad

                if wd != 0.0:
                    g = g + wd * p.data

                st = self._get_state(p, amsgrad)
                st["step"] += 1
                t = st["step"]

                # Update biased moment estimates
                st["m"] = beta1 * st["m"] + (1.0 - beta1) * g
                st["v"] = beta2 * st["v"] + (1.0 - beta2) * g * g

                # Bias-corrected estimates
                m_hat = st["m"] / (1.0 - beta1 ** t)
                v_hat = st["v"] / (1.0 - beta2 ** t)

                if amsgrad:
                    st["v_max"] = max(st["v_max"], v_hat)
                    denom = math.sqrt(st["v_max"]) + eps
                else:
                    denom = math.sqrt(v_hat) + eps

                p.data -= lr * m_hat / denom

        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        b1, b2 = g["betas"]
        return (
            f"Adam(lr={g['lr']}, betas=({b1},{b2}), eps={g['eps']}, "
            f"weight_decay={g['weight_decay']}, amsgrad={g['amsgrad']}, "
            f"params={sum(len(g['params']) for g in self.param_groups)})"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  AdamW — Adam with decoupled weight decay                                  #
# ═══════════════════════════════════════════════════════════════════════════ #


class AdamW(Adam):
    """
    Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2019).

    The key insight: in Adam, adding λ·p to the gradient before the update
    couples the decay to the adaptive step size 1/√v̂, making the effective
    decay smaller for parameters with large gradient variance. This is not
    L2 regularisation — it biases the optimizer toward directions of low
    curvature.

    AdamW fixes this by applying decay directly to the parameter, *after*
    the adaptive gradient step, so the decay magnitude is always lr·λ·p
    regardless of gradient statistics:

        p_t = p_{t-1} · (1 - lr·λ)  - lr · m̂_t / (√v̂_t + ε)

    This is the default optimizer for most modern transformer training.

    Parameters
    ----------
    Same as Adam, but weight_decay now refers to *decoupled* decay λ.
    weight_decay=0.01 is the recommended default (not 0.0).
    """

    def __init__(
        self,
        parameters,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,   # non-zero default — this is the point of AdamW
        amsgrad: bool = False,
    ):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps,
                         weight_decay=0.0,   # suppress Adam's coupled decay
                         amsgrad=amsgrad)
        # Store decoupled decay separately so Adam's g += wd*p path is bypassed
        for group in self.param_groups:
            group["decoupled_wd"] = weight_decay

    def add_param_group(self, param_group):
        param_group.setdefault("decoupled_wd", 0.01)
        super().add_param_group(param_group)

    def step(self):
        self._maybe_clip()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["decoupled_wd"]
            amsgrad      = group["amsgrad"]

            for p in group["params"]:
                g = p.grad   # raw gradient — no L2 added here

                st = self._get_state(p, amsgrad)
                st["step"] += 1
                t = st["step"]

                st["m"] = beta1 * st["m"] + (1.0 - beta1) * g
                st["v"] = beta2 * st["v"] + (1.0 - beta2) * g * g

                m_hat = st["m"] / (1.0 - beta1 ** t)
                v_hat = st["v"] / (1.0 - beta2 ** t)

                if amsgrad:
                    st["v_max"] = max(st["v_max"], v_hat)
                    denom = math.sqrt(st["v_max"]) + eps
                else:
                    denom = math.sqrt(v_hat) + eps

                # ── Decoupled decay applied FIRST, then adaptive gradient step ──
                # This ensures decay magnitude = lr·λ·p, independent of curvature
                p.data *= (1.0 - lr * wd)
                p.data -= lr * m_hat / denom

        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        b1, b2 = g["betas"]
        return (
            f"AdamW(lr={g['lr']}, betas=({b1},{b2}), eps={g['eps']}, "
            f"weight_decay={g.get('decoupled_wd', 0.01)}, "
            f"params={sum(len(g['params']) for g in self.param_groups)})"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Learning Rate Schedulers                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #


class _LRScheduler:
    """
    Base class for learning rate schedulers.

    Schedulers maintain a reference to the optimizer and call
    `optimizer.lr = new_lr` on each `.step()`.

    Design note: schedulers operate on *optimizer steps*, not epochs —
    this makes their behaviour deterministic regardless of dataset size.
    """

    def __init__(self, optimizer: Optimizer, last_step: int = -1):
        self.optimizer = optimizer
        self.last_step = last_step
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        # Apply initial lr at construction
        if last_step == -1:
            self.step()

    def get_lr(self) -> list[float]:
        """Return new lr for each param group. Override in subclasses."""
        raise NotImplementedError

    def step(self):
        self.last_step += 1
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def __repr__(self):
        return f"{self.__class__.__name__}(step={self.last_step})"


class StepLR(_LRScheduler):
    """
    Multiply lr by γ every `step_size` optimizer steps.

    lr_t = lr_0 · γ^⌊t / step_size⌋

    Parameters
    ----------
    optimizer  : Optimizer
    step_size  : int   — steps between each decay
    gamma      : float — multiplicative factor (< 1 to decay, > 1 to grow)
    last_step  : int   — step to resume from (-1 = fresh start)
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1,
                 last_step: int = -1):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list[float]:
        decay = self.gamma ** (self.last_step // self.step_size)
        return [base * decay for base in self.base_lrs]

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma}, step={self.last_step})"


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing from lr_max → lr_min over T_max steps (SGDR-style).

    lr_t = lr_min + ½ · (lr_max - lr_min) · (1 + cos(π · t / T_max))

    At t=0:     lr = lr_max  (cos(0) = 1)
    At t=T_max: lr = lr_min  (cos(π) = -1)
    After T_max: lr stays at lr_min (no automatic restart here —
                 call restart() to reset t=0 with a new lr_max if desired).

    Parameters
    ----------
    optimizer  : Optimizer
    T_max      : int   — number of steps for one cosine cycle
    eta_min    : float — minimum lr at end of cycle (default 0)
    last_step  : int   — step to resume from
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0,
                 last_step: int = -1):
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list[float]:
        t = min(self.last_step, self.T_max)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))
        return [
            self.eta_min + (base - self.eta_min) * cos_factor
            for base in self.base_lrs
        ]

    def restart(self, new_lr_max: float | None = None):
        """Reset cycle to t=0, optionally with a new lr_max (warm restart)."""
        self.last_step = -1
        if new_lr_max is not None:
            self.base_lrs = [new_lr_max] * len(self.base_lrs)
        self.step()

    def __repr__(self):
        return (f"CosineAnnealingLR(T_max={self.T_max}, eta_min={self.eta_min}, "
                f"step={self.last_step})")


class LinearWarmupLR(_LRScheduler):
    """
    Linear warmup for `warmup_steps` steps, then delegate to a wrapped scheduler.

    lr_t = base_lr · (t+1) / warmup_steps     if t < warmup_steps
         = wrapped_scheduler.get_lr()          otherwise

    Warmup prevents the large gradient variance at initialisation from
    driving Adam's moment estimates in a bad direction before they've
    stabilised. The first warmup_steps steps see lr ramp from
    base_lr/warmup_steps → base_lr linearly.

    Parameters
    ----------
    optimizer      : Optimizer
    warmup_steps   : int          — steps to ramp up over
    after_scheduler: _LRScheduler — scheduler to hand off to after warmup
                                    (pass None to hold at base_lr)
    last_step      : int
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 after_scheduler: "_LRScheduler | None" = None,
                 last_step: int = -1):
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list[float]:
        t = self.last_step

        if t < self.warmup_steps:
            # Linear ramp: step 0 → base/warmup_steps, step warmup_steps-1 → base
            scale = (t + 1) / self.warmup_steps
            return [base * scale for base in self.base_lrs]

        if self.after_scheduler is not None:
            # Advance the wrapped scheduler's internal step
            # (it was constructed at -1, so offset by warmup)
            self.after_scheduler.last_step = t - self.warmup_steps
            return self.after_scheduler.get_lr()

        return list(self.base_lrs)   # hold at base_lr

    def __repr__(self):
        return (f"LinearWarmupLR(warmup_steps={self.warmup_steps}, "
                f"after={self.after_scheduler}, step={self.last_step})")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Convenience factory                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #


def get_optimizer(name: str, parameters, **kwargs) -> Optimizer:
    """
    Factory function. Returns an optimizer by name string.

    Supported names (case-insensitive):
        'sgd', 'adagrad', 'rmsprop', 'adam', 'adamw'

    Example
    -------
        opt = get_optimizer('adamw', model.parameters(), lr=3e-4, weight_decay=0.01)
    """
    registry = {
        "sgd": SGD,
        "adagrad": AdaGrad,
        "rmsprop": RMSProp,
        "adam": Adam,
        "adamw": AdamW,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {list(registry)}")
    return registry[key](parameters, **kwargs)