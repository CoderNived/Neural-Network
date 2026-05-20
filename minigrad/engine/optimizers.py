import math

def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
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

class Optimizer:
    def __init__(self, parameters, defaults: dict):
        self.param_groups = []
        self._step_count = 0
        self._clip_norm = None
        self._clip_norm_type = 2.0
        params = list(parameters)
        if not params:
            raise ValueError("optimizer received an empty parameter list")
        if "lr" in defaults and defaults["lr"] <= 0:
            raise ValueError(f"Learning rate must be positive, got {defaults['lr']}")
        if "weight_decay" in defaults and defaults["weight_decay"] < 0:
            raise ValueError(f"weight_decay must be non-negative")
        self.add_param_group({"params": params, **defaults})

    def add_param_group(self, param_group: dict):
        if "params" not in param_group:
            raise ValueError("param_group must contain a 'params' key")
        param_group = dict(param_group)
        param_group["params"] = list(param_group["params"])
        self.param_groups.append(param_group)
        self._init_group(param_group)

    def _init_group(self, group: dict):
        pass

    def set_grad_clip(self, max_norm: float, norm_type: float = 2.0):
        self._clip_norm = max_norm
        self._clip_norm_type = norm_type
        return self

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = 0.0

    def step(self):
        raise NotImplementedError

    def _maybe_clip(self):
        if self._clip_norm is not None:
            all_params = [p for g in self.param_groups for p in g["params"]]
            return clip_grad_norm_(all_params, self._clip_norm, self._clip_norm_type)
        return None

    @property
    def lr(self):
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value):
        self.param_groups[0]["lr"] = value

    def state_dict(self) -> dict:
        return {
            "step_count": self._step_count,
            "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups],
            "state": getattr(self, "state", {}),
        }

    def load_state_dict(self, d: dict):
        self._step_count = d["step_count"]
        for g, saved in zip(self.param_groups, d["param_groups"]):
            g.update(saved)
        if hasattr(self, "state"):
            self.state.update(d.get("state", {}))


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0, dampening=0.0):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if nesterov and (momentum == 0.0 or dampening != 0.0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, dampening=dampening)
        super().__init__(parameters, defaults)

    def _init_group(self, group):
        group["velocities"] = [0.0] * len(group["params"])

    def step(self):
        self._maybe_clip()
        for group in self.param_groups:
            lr = group["lr"]; beta = group["momentum"]
            nesterov = group["nesterov"]; wd = group["weight_decay"]
            dampening = group["dampening"]; velocities = group["velocities"]
            for i, p in enumerate(group["params"]):
                g = p.grad
                if wd != 0.0:
                    g = g + wd * p.data
                if beta != 0.0:
                    v = beta * velocities[i] + (1.0 - dampening) * g
                    velocities[i] = v
                    g = beta * v + g if nesterov else v
                p.data -= lr * g
        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        return f"SGD(lr={g['lr']}, momentum={g['momentum']}, params={sum(len(g['params']) for g in self.param_groups)})"


class AdaGrad(Optimizer):
    def __init__(self, parameters, lr=0.01, eps=1e-10, weight_decay=0.0, lr_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, lr_decay=lr_decay)
        super().__init__(parameters, defaults)

    def _init_group(self, group):
        group["sum_sq"] = [0.0] * len(group["params"])

    def step(self):
        self._maybe_clip()
        for group in self.param_groups:
            lr = group["lr"]; eps = group["eps"]
            wd = group["weight_decay"]; lr_decay = group["lr_decay"]
            sum_sq = group["sum_sq"]
            clr = lr / (1.0 + self._step_count * lr_decay)
            for i, p in enumerate(group["params"]):
                g = p.grad + (group["weight_decay"] * p.data if wd else 0)
                sum_sq[i] += g * g
                p.data -= clr * g / (math.sqrt(sum_sq[i]) + eps)
        self._step_count += 1


class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8,
                 momentum=0.0, weight_decay=0.0, centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum,
                        weight_decay=weight_decay, centered=centered)
        super().__init__(parameters, defaults)

    def _init_group(self, group):
        n = len(group["params"])
        group["sq_avg"] = [0.0] * n
        group["momentum_buf"] = [0.0] * n
        if group["centered"]:
            group["grad_avg"] = [0.0] * n

    def step(self):
        self._maybe_clip()
        for group in self.param_groups:
            lr = group["lr"]; alpha = group["alpha"]; eps = group["eps"]
            mom = group["momentum"]; wd = group["weight_decay"]
            centered = group["centered"]
            sq_avg = group["sq_avg"]; mbuf = group["momentum_buf"]
            grad_avg = group.get("grad_avg", None)
            for i, p in enumerate(group["params"]):
                g = p.grad + (wd * p.data if wd else 0)
                sq_avg[i] = alpha * sq_avg[i] + (1.0 - alpha) * g * g
                if centered:
                    grad_avg[i] = alpha * grad_avg[i] + (1.0 - alpha) * g
                    denom = math.sqrt(sq_avg[i] - grad_avg[i]**2) + eps
                else:
                    denom = math.sqrt(sq_avg[i]) + eps
                if mom != 0.0:
                    mbuf[i] = mom * mbuf[i] + g / denom
                    p.data -= lr * mbuf[i]
                else:
                    p.data -= lr * g / denom
        self._step_count += 1


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, amsgrad=False):
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(parameters, defaults)
        self.state = {}

    def _get_state(self, p, amsgrad):
        pid = id(p)
        if pid not in self.state:
            self.state[pid] = {"step": 0, "m": 0.0, "v": 0.0,
                               "v_max": 0.0 if amsgrad else None}
        return self.state[pid]

    def step(self):
        self._maybe_clip()
        for group in self.param_groups:
            lr = group["lr"]; beta1, beta2 = group["betas"]
            eps = group["eps"]; wd = group["weight_decay"]; amsgrad = group["amsgrad"]
            for p in group["params"]:
                g = p.grad + (wd * p.data if wd else 0)
                st = self._get_state(p, amsgrad)
                st["step"] += 1; t = st["step"]
                st["m"] = beta1 * st["m"] + (1.0 - beta1) * g
                st["v"] = beta2 * st["v"] + (1.0 - beta2) * g * g
                m_hat = st["m"] / (1.0 - beta1**t)
                v_hat = st["v"] / (1.0 - beta2**t)
                if amsgrad:
                    st["v_max"] = max(st["v_max"], v_hat)
                    denom = math.sqrt(st["v_max"]) + eps
                else:
                    denom = math.sqrt(v_hat) + eps
                p.data -= lr * m_hat / denom
        self._step_count += 1

    def __repr__(self):
        g = self.param_groups[0]
        return f"Adam(lr={g['lr']}, params={sum(len(g['params']) for g in self.param_groups)})"


class AdamW(Adam):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01, amsgrad=False):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps,
                         weight_decay=0.0, amsgrad=amsgrad)
        for group in self.param_groups:
            group["decoupled_wd"] = weight_decay

    def step(self):
        self._maybe_clip()
        for group in self.param_groups:
            lr = group["lr"]; beta1, beta2 = group["betas"]
            eps = group["eps"]; wd = group["decoupled_wd"]; amsgrad = group["amsgrad"]
            for p in group["params"]:
                g = p.grad
                st = self._get_state(p, amsgrad)
                st["step"] += 1; t = st["step"]
                st["m"] = beta1 * st["m"] + (1.0 - beta1) * g
                st["v"] = beta2 * st["v"] + (1.0 - beta2) * g * g
                m_hat = st["m"] / (1.0 - beta1**t)
                v_hat = st["v"] / (1.0 - beta2**t)
                if amsgrad:
                    st["v_max"] = max(st["v_max"], v_hat)
                    denom = math.sqrt(st["v_max"]) + eps
                else:
                    denom = math.sqrt(v_hat) + eps
                p.data *= (1.0 - lr * wd)
                p.data -= lr * m_hat / denom
        self._step_count += 1


class _LRScheduler:
    def __init__(self, optimizer, last_step=-1):
        self.optimizer = optimizer
        self.last_step = last_step
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        if last_step == -1:
            self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_step += 1
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_step=-1):
        self.step_size = step_size; self.gamma = gamma
        super().__init__(optimizer, last_step)

    def get_lr(self):
        decay = self.gamma ** (self.last_step // self.step_size)
        return [base * decay for base in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_step=-1):
        self.T_max = T_max; self.eta_min = eta_min
        super().__init__(optimizer, last_step)

    def get_lr(self):
        t = min(self.last_step, self.T_max)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))
        return [self.eta_min + (base - self.eta_min) * cos_factor for base in self.base_lrs]


class LinearWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, after_scheduler=None, last_step=-1):
        self.warmup_steps = warmup_steps; self.after_scheduler = after_scheduler
        super().__init__(optimizer, last_step)

    def get_lr(self):
        t = self.last_step
        if t < self.warmup_steps:
            return [base * (t + 1) / self.warmup_steps for base in self.base_lrs]
        if self.after_scheduler is not None:
            self.after_scheduler.last_step = t - self.warmup_steps
            return self.after_scheduler.get_lr()
        return list(self.base_lrs)


def get_optimizer(name: str, parameters, **kwargs):
    registry = {"sgd": SGD, "adagrad": AdaGrad, "rmsprop": RMSProp, "adam": Adam, "adamw": AdamW}
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {list(registry)}")
    return registry[key](parameters, **kwargs)