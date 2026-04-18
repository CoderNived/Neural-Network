import math

# ─────────────────────────────────────────
# VECTOR OPERATIONS
# ─────────────────────────────────────────

def dot(a, b):
    """
    Geometric meaning: measures how much a and b point in the same direction.
    dot = ||a|| * ||b|| * cos(theta)
    Zero -> orthogonal; maximum when parallel.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))

def vec_add(a, b):
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return [x + y for x, y in zip(a, b)]

def scalar_mul(v, s):
    return [x * s for x in v]


# ─────────────────────────────────────────
# MATRIX REPRESENTATION
# ─────────────────────────────────────────
#
# Choice: list of rows, where each row is a list of floats.
# A[i][j] -> row i, column j. Natural Python indexing.
#
# Cache performance: row access is contiguous (good).
# Column access requires striding across rows (cache-unfriendly) --
# same tradeoff as NumPy C-order. For our triple loop we iterate
# over rows of A and rows of B^T (= columns of B), so we transpose B
# once upfront to make the inner loop cache-friendly.
#
# Transpose cost: O(m*n) -- must build a new list-of-rows.
# We always deep-copy to keep original and transpose independent.

def shape(A):
    return (len(A), len(A[0]))

def _check_rect(A, name="Matrix"):
    cols = len(A[0])
    for row in A:
        if len(row) != cols:
            raise ValueError(f"{name} is not rectangular")


# ─────────────────────────────────────────
# MATRIX OPERATIONS
# ─────────────────────────────────────────

def transpose(A):
    """
    Returns a completely independent copy -- no shared references.
    Mutation of A will not corrupt the result.
    """
    m, n = shape(A)
    # Build new rows from columns of A
    return [[A[i][j] for i in range(m)] for j in range(n)]

def mat_vec_mul(A, v):
    m, n = shape(A)
    if len(v) != n:
        raise ValueError(
            f"Shape mismatch: matrix is ({m}x{n}), vector has length {len(v)}"
        )
    return [dot(row, v) for row in A]

def mat_mul(A, B):
    """
    C[i][j] = dot(row i of A, col j of B)
    = the composed linear transformation: apply B, then A.
    
    We transpose B once so inner loop accesses rows (cache-friendly).
    """
    _check_rect(A, "A")
    _check_rect(B, "B")

    m, k1 = shape(A)
    k2, n = shape(B)

    if k1 != k2:
        raise ValueError(
            f"Shape mismatch for mat_mul: ({m}x{k1}) @ ({k2}x{n}) -- "
            f"inner dimensions {k1} != {k2}"
        )

    Bt = transpose(B)  # Bt[j] is column j of B -- now a row, cache-friendly
    return [
        [dot(A[i], Bt[j]) for j in range(n)]
        for i in range(m)
    ]


# ─────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ─────────────────────────────────────────

def relu(x):
    """Scalar relu."""
    return max(0.0, x)

def relu_vec(v):
    return [relu(x) for x in v]


def sigmoid(x):
    """
    Numerically stable sigmoid.
    
    Naive form: 1 / (1 + e^{-x})
    Problem: e^{-x} overflows to inf for large negative x -> 0/0 territory.
    
    Stable form:
      x >= 0:  1 / (1 + e^{-x})          -- e^{-x} is small, safe
      x <  0:  e^x / (1 + e^x)           -- e^x is small, safe
    
    Both are mathematically equivalent (multiply top & bottom by e^x),
    but each branch only computes a small exponential.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def sigmoid_vec(v):
    return [sigmoid(x) for x in v]


def tanh_scalar(x):
    """
    tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    
    Derived from: tanh(x) = 2*sigmoid(2x) - 1  (useful identity)
    But we derive directly from exponentials here.
    
    Numerically stable: rewrite as
      x >= 0:  let u = e^{-2x};  (1 - u) / (1 + u)
      x <  0:  let u = e^{2x};   (u - 1) / (u + 1)
    """
    if x >= 0:
        u = math.exp(-2 * x)
        return (1.0 - u) / (1.0 + u)
    else:
        u = math.exp(2 * x)
        return (u - 1.0) / (u + 1.0)

def tanh_vec(v):
    return [tanh_scalar(x) for x in v]


# ─────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────

def approx_eq(a, b, tol=1e-9):
    return abs(a - b) < tol

def mat_approx_eq(A, B, tol=1e-9):
    if shape(A) != shape(B):
        return False
    return all(
        approx_eq(A[i][j], B[i][j], tol)
        for i in range(len(A))
        for j in range(len(A[0]))
    )

def run_tests():
    errors = []

    # ── dot product ──────────────────────────────
    assert dot([1,0],[0,1]) == 0,       "orthogonal vectors -> 0"
    assert dot([1,0],[1,0]) == 1,       "parallel unit vectors -> 1"
    assert dot([3,4],[3,4]) == 25,      "||v||^2"
    try:
        dot([1,2],[1])
        errors.append("dot: should raise on length mismatch")
    except ValueError:
        pass

    # ── vec_add ──────────────────────────────────
    assert vec_add([1,2,3],[4,5,6]) == [5,7,9]

    # ── scalar_mul ───────────────────────────────
    assert scalar_mul([1,2,3], 2) == [2,4,6]

    # ── transpose ────────────────────────────────
    A = [[1,2,3],[4,5,6]]
    At = transpose(A)
    assert shape(At) == (3,2)
    assert At[0] == [1,4]
    assert At[2] == [3,6]
    # Independence check
    A[0][0] = 999
    assert At[0][0] == 1, "transpose must be independent copy"
    A[0][0] = 1  # restore

    # ── mat_mul: identity ─────────────────────────
    I = [[1,0],[0,1]]
    M = [[3,4],[5,6]]
    assert mat_approx_eq(mat_mul(I, M), M), "I @ M == M"
    assert mat_approx_eq(mat_mul(M, I), M), "M @ I == M"

    # ── mat_mul: (2x3) @ (3x2) -> (2x2) ──────────
    M1 = [[1,2,3],[4,5,6]]
    M2 = [[7,8],[9,10],[11,12]]
    result = mat_mul(M1, M2)
    assert shape(result) == (2,2)
    assert result[0][0] == 58,  f"got {result[0][0]}"
    assert result[0][1] == 64,  f"got {result[0][1]}"
    assert result[1][0] == 139, f"got {result[1][0]}"
    assert result[1][1] == 154, f"got {result[1][1]}"

    # ── mat_mul: associativity (AB)C == A(BC) ─────
    A2 = [[1,2],[3,4]]
    B2 = [[0,1],[1,0]]
    C2 = [[2,0],[0,2]]
    assert mat_approx_eq(mat_mul(mat_mul(A2,B2),C2), mat_mul(A2,mat_mul(B2,C2))), \
        "Associativity failed"

    # ── mat_mul: shape mismatch ───────────────────
    try:
        mat_mul([[1,2],[3,4],[5,6]], [[1,2],[3,4],[5,6]])  # (3x2)@(3x2)
        errors.append("mat_mul: should raise on shape mismatch")
    except ValueError:
        pass

    # ── mat_vec_mul ──────────────────────────────
    A3 = [[1,0],[0,1]]
    assert mat_vec_mul(A3, [3,5]) == [3,5], "identity mat_vec_mul"

    # ── sigmoid ───────────────────────────────────
    assert approx_eq(sigmoid(0), 0.5),          f"sigmoid(0) = {sigmoid(0)}"
    assert approx_eq(sigmoid(100), 1.0, 1e-6),  f"sigmoid(100) = {sigmoid(100)}"
    assert approx_eq(sigmoid(-100), 0.0, 1e-6), f"sigmoid(-100) = {sigmoid(-100)}"
    # Stability: these must not raise OverflowError
    sigmoid(1000)
    sigmoid(-1000)

    # ── relu ──────────────────────────────────────
    assert relu(3.0)  == 3.0
    assert relu(-3.0) == 0.0
    assert relu(0.0)  == 0.0

    # ── tanh ──────────────────────────────────────
    assert approx_eq(tanh_scalar(0), 0.0)
    assert approx_eq(tanh_scalar(100), 1.0, 1e-6)
    assert approx_eq(tanh_scalar(-100), -1.0, 1e-6)
    assert approx_eq(tanh_scalar(0.5), math.tanh(0.5), 1e-12), \
        f"tanh mismatch at 0.5: {tanh_scalar(0.5)} vs {math.tanh(0.5)}"

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        raise AssertionError("Some tests failed")
    else:
        print("All tests passed.")

run_tests()