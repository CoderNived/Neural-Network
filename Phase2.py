class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None   # no-op by default
        self._parents = set(_parents)   # set, not list:
                                        # order doesn't matter here,
                                        # and set prevents duplicate edges
                                        # if somehow the same Value appears twice
        self._op = _op                  # for debugging only

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ─────────────────────────────────────────
    # FORWARD OPERATIONS
    # ─────────────────────────────────────────

    def __add__(self, other):
        # d(a+b)/da = 1,  d(a+b)/db = 1
        # Gradient flows through addition unchanged — it's a "splitter"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _parents=(self, other), _op='+')

        def _backward():
            # += because self or other may be reused elsewhere in the graph
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        # d(a*b)/da = b,  d(a*b)/db = a
        # The backward needs the *forward* values of a and b.
        # They're captured in the closure here, at the moment of the
        # forward pass — this is why closures are the right tool.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _parents=(self, other), _op='*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    # ─────────────────────────────────────────
    # BACKWARD PASS
    # ─────────────────────────────────────────

    def backward(self):
        # Why are computation graphs always acyclic?
        # Because every operation creates a *new* Value object.
        # A node's parents always existed before it did.
        # You cannot create a cycle without time travel —
        # a node cannot be its own ancestor in a single forward pass.

        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:        # id() handles duplicate Value(same number)
                visited.add(id(v))
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)              # post-order: node appended after its parents

        build_topo(self)
        # topo is now leaves-first, output-last
        # Reverse -> output-first: each node runs _backward
        # only after all its consumers have already sent gradient to it

        self.grad = 1.0                     # d(loss)/d(loss) = 1

        for node in reversed(topo):
            node._backward()


# ─────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────

def approx_eq(a, b, tol=1e-9):
    return abs(a - b) < tol

def run_tests():

    # Test 1 — Addition
    a = Value(3.0); b = Value(4.0)
    c = a + b
    c.backward()
    assert approx_eq(a.grad, 1.0), f"T1 a.grad: {a.grad}"
    assert approx_eq(b.grad, 1.0), f"T1 b.grad: {b.grad}"

    # Test 2 — Multiplication
    a = Value(3.0); b = Value(4.0)
    c = a * b
    c.backward()
    assert approx_eq(a.grad, 4.0), f"T2 a.grad: {a.grad}"  # dc/da = b = 4
    assert approx_eq(b.grad, 3.0), f"T2 b.grad: {b.grad}"  # dc/db = a = 3

    # Test 3 — Composed ops, two paths through a
    a = Value(2.0); b = Value(3.0)
    c = a * b       # c = 6
    d = c + a       # d = 8
    d.backward()
    # a.grad: path via c (1 * b = 3.0) + direct path (1.0) = 4.0
    # b.grad: path via c only: 1 * a = 2.0
    assert approx_eq(a.grad, 4.0), f"T3 a.grad: {a.grad}"
    assert approx_eq(b.grad, 2.0), f"T3 b.grad: {b.grad}"

    # Test 4 — Gradient accumulation (explicitly catches = vs +=)
    # Simplest graph where overwrite breaks things: x used twice
    x = Value(3.0)
    y = x + x       # y = 6, dy/dx = 2 (two paths, each contributing 1)
    y.backward()
    assert approx_eq(x.grad, 2.0), f"T4 x.grad: {x.grad} (expected 2.0 — catches = vs +=)"

    # Test 5 — Topological order violation detector
    # Build a diamond: a -> b, a -> c, b+c -> d
    # If order is wrong, d.grad won't reach a correctly
    a = Value(2.0)
    b = a * Value(3.0)   # b = 6
    c = a * Value(4.0)   # c = 8
    d = b + c            # d = 14
    d.backward()
    # dd/da = dd/db * db/da + dd/dc * dc/da = 1*3 + 1*4 = 7
    assert approx_eq(a.grad, 7.0), f"T5 a.grad: {a.grad} (diamond graph)"

    # Test 6 — Scalar interop (Value + plain number)
    a = Value(3.0)
    b = a + 2       # should work via __add__ wrapping
    b.backward()
    assert approx_eq(a.grad, 1.0), f"T6 a.grad: {a.grad}"

    # Test 7 — Longer chain, verify gradient flows all the way back
    a = Value(2.0)
    b = a * Value(3.0)   # b = 6
    c = b * Value(4.0)   # c = 24
    c.backward()
    # dc/da = dc/db * db/da = 4 * 3 = 12
    assert approx_eq(a.grad, 12.0), f"T7 a.grad: {a.grad}"

    print("All tests passed.")

run_tests()