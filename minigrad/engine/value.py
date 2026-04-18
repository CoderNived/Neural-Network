"""
engine/value.py
---------------
Scalar-level reverse-mode automatic differentiation engine for minigrad.
No NumPy, no external libraries — standard library only.

Design:
    Every scalar in a computation is wrapped in a Value.
    Each Value stores:
        data       — the forward-pass scalar
        grad       — accumulated gradient (d_loss / d_self)
        _backward  — closure that pushes gradient to parents
        _parents   — set of Values that produced this one
        _op        — string label for debugging / visualisation

    Calling .backward() on the final output node:
        1. Topologically sorts the DAG (post-order DFS)
        2. Sets self.grad = 1.0  (d_loss/d_loss = 1)
        3. Calls _backward() on every node in reverse topological order
           so each node receives its full accumulated gradient before
           it propagates to its parents.

Why a set for _parents?
    Order doesn't matter (we sort separately).
    A set makes the "node appears twice" case explicit — but note: in
    practice  x + x  creates two references to the same object, not a
    set with one element, which is why we use id() in the topo sort
    rather than object identity in the set.  The set still deduplicates
    genuinely identical parent objects in ops like  a * 1 + a * 1.

Why closures for _backward?
    The backward of a * b needs the *forward* values of a and b.
    Closures capture them at the moment the operation is built,
    before any weights are updated.  This is exactly what PyTorch's
    saved_tensors mechanism does.

Why are computation graphs always acyclic?
    Every operation returns a *new* Value.  A node's parents always
    pre-exist it.  You cannot form a cycle without re-using an output
    as its own input within the same forward pass, which the API
    makes impossible — there is no "assign" operation.
"""

import math


class Value:

    def __init__(self, data, _parents=(), _op='', _label=''):
        self.data      = float(data)
        self.grad      = 0.0
        self._backward = lambda: None
        self._parents  = set(_parents)
        self._op       = _op        # for debug / graph visualisation
        self._label    = _label     # optional human-readable name

    def __repr__(self):
        return (f"Value(data={self.data:.6g}, grad={self.grad:.6g}"
                + (f", op='{self._op}'" if self._op else "")
                + (f", label='{self._label}'" if self._label else "")
                + ")")

    # ═══════════════════════════════════════════════════════
    # FORWARD OPERATIONS
    # ═══════════════════════════════════════════════════════

    # ── Addition ─────────────────────────────────────────

    def __add__(self, other):
        """
        out = self + other
        d(out)/d(self)  = 1
        d(out)/d(other) = 1

        Addition is a gradient *distributor*: it passes the incoming
        gradient unchanged to both parents.  Intuitively, if the output
        increases by δ, both inputs contributed equally to that increase.
        """
        other = _ensure_value(other)
        out   = Value(self.data + other.data, _parents=(self, other), _op='+')

        def _backward():
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):   # other + self
        return self.__add__(other)

    # ── Multiplication ───────────────────────────────────

    def __mul__(self, other):
        """
        out = self * other
        d(out)/d(self)  = other.data   (captured in closure)
        d(out)/d(other) = self.data    (captured in closure)

        The backward closure captures the *forward* values of self and
        other.  This is critical: by the time _backward() is called,
        self.data and other.data still hold the values from the forward
        pass, so the gradients are computed correctly even after weights
        are updated (because we create a new graph on each forward pass).
        """
        other = _ensure_value(other)
        out   = Value(self.data * other.data, _parents=(self, other), _op='*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):   # other * self
        return self.__mul__(other)

    # ── Subtraction & Negation ───────────────────────────

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_ensure_value(other))

    def __rsub__(self, other):   # other - self
        return _ensure_value(other) + (-self)

    # ── Division ─────────────────────────────────────────

    def __truediv__(self, other):
        """
        out = self / other  =  self * other^{-1}
        Implemented as multiplication by the reciprocal so the backward
        is derived automatically from __mul__ and __pow__.
        """
        return self * _ensure_value(other) ** -1

    def __rtruediv__(self, other):   # other / self
        return _ensure_value(other) * self ** -1

    # ── Power ────────────────────────────────────────────

    def __pow__(self, exponent):
        """
        out = self ^ exponent       (exponent is a plain Python number)
        d(out)/d(self) = exponent * self^(exponent - 1)

        Restriction: exponent must be int or float, not a Value.
        Supporting Value exponents would require log in the backward,
        which is added separately in exp() / log().
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                f"__pow__ exponent must be int or float, got {type(exponent)}. "
                "Use exp() and log() for Value exponents."
            )
        out = Value(self.data ** exponent, _parents=(self,), _op=f'**{exponent}')

        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # ── Exponential & Logarithm ──────────────────────────

    def exp(self):
        """
        out = e^self
        d(out)/d(self) = e^self = out.data

        Note: out.data is captured in the closure, not recomputed.
        This is the standard trick — exp is its own derivative.
        """
        out = Value(math.exp(self.data), _parents=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        out = ln(self)
        d(out)/d(self) = 1 / self.data

        Guard: self.data must be > 0.  A log of zero or negative is a
        sign that your network has diverged or your loss is ill-formed.
        """
        if self.data <= 0:
            raise ValueError(
                f"log() called on non-positive value: {self.data}. "
                "Check for dead neurons or exploding activations."
            )
        out = Value(math.log(self.data), _parents=(self,), _op='log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ── Activation Functions ─────────────────────────────

    def relu(self):
        """
        out = max(0, self)
        d(out)/d(self) = 1 if self.data > 0 else 0

        Sub-gradient convention: gradient at exactly 0 is 0.
        """
        out = Value(max(0.0, self.data), _parents=(self,), _op='relu')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        out = σ(self) = 1 / (1 + e^{-self})
        d(out)/d(self) = σ(self) * (1 - σ(self)) = out.data * (1 - out.data)

        Numerically stable: uses the same two-branch form as linalg.py.
        The backward reuses out.data (already computed), avoiding a
        second call to exp.
        """
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, _parents=(self,), _op='sigmoid')

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        out = tanh(self)
        d(out)/d(self) = 1 - tanh²(self) = 1 - out.data²

        Numerically stable two-branch form; backward reuses out.data.
        """
        x = self.data
        if x >= 0:
            u = math.exp(-2.0 * x)
            t = (1.0 - u) / (1.0 + u)
        else:
            u = math.exp(2.0 * x)
            t = (u - 1.0) / (u + 1.0)
        out = Value(t, _parents=(self,), _op='tanh')

        def _backward():
            self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    # ═══════════════════════════════════════════════════════
    # BACKWARD PASS  (reverse-mode autodiff)
    # ═══════════════════════════════════════════════════════

    def backward(self):
        """
        Compute gradients for all nodes in the computation graph via
        reverse-mode automatic differentiation.

        Algorithm:
            1. Build topological order with post-order DFS.
               Post-order guarantees: a node is appended only after
               ALL of its parents have been appended.
               → topo[-1] is always self (the output / loss node).
               → topo[0] is always a leaf with no parents.

            2. Set self.grad = 1.0
               (d_loss / d_loss = 1, the seed gradient)

            3. Walk topo in reverse (output → leaves).
               At each node, _backward() distributes the node's
               accumulated grad to its parents via +=.
               By the time we reach a parent node, all of its
               consumers have already sent their contributions.
               → Every node receives its *complete* gradient
                 before it propagates further.

        Data structure: plain Python list used as a stack.
            - DFS uses the call stack (recursive).
            - visited uses id() not the object itself, because two
              Value(3.0) objects are distinct nodes that happen to
              hold the same number.  Using the object as a set element
              would rely on default __hash__ (id-based), which works,
              but being explicit avoids confusion if __hash__ is ever
              overridden.
        """
        topo    = []
        visited = set()

        def _build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for parent in v._parents:
                    _build_topo(parent)
                topo.append(v)   # post-order: appended after all parents

        _build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):   # output-first, leaves-last
            node._backward()

    def zero_grad(self):
        """Reset gradients for this node and all ancestors."""
        visited = set()

        def _zero(v):
            if id(v) not in visited:
                visited.add(id(v))
                v.grad = 0.0
                for parent in v._parents:
                    _zero(parent)

        _zero(self)

    # ═══════════════════════════════════════════════════════
    # GRAPH UTILITIES
    # ═══════════════════════════════════════════════════════

    def topo_order(self):
        """Return nodes in topological order (leaves first, self last)."""
        topo    = []
        visited = set()

        def _build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._parents:
                    _build(p)
                topo.append(v)

        _build(self)
        return topo

    def graph_summary(self):
        """Print a compact summary of the computation graph."""
        nodes = self.topo_order()
        print(f"Graph: {len(nodes)} nodes")
        for n in reversed(nodes):
            parents_str = ", ".join(
                f"Value({p.data:.4g})" for p in n._parents
            )
            print(f"  {n!r}  ← [{parents_str}]")


# ═══════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════

def _ensure_value(x):
    """Wrap a plain number in Value if needed."""
    return x if isinstance(x, Value) else Value(x)