"""
engine/linalg.py
----------------
Pure-Python linear algebra engine for minigrad.
No NumPy, no external libraries — standard library only.

Design decisions:
  - Matrices are list-of-rows: A[i][j] = row i, col j.
  - Row access is contiguous (cache-friendly); column access is not.
  - mat_mul transposes B once upfront so the inner dot loop is row-access only.
  - All operations return new objects — no in-place mutation unless stated.
  - Every public function validates shapes and raises ValueError on mismatch.
"""

import math
from typing import List

# Type alias for readability
Matrix = List[List[float]]
Vector = List[float]


# ═══════════════════════════════════════════════════════════
# INTERNAL UTILITIES
# ═══════════════════════════════════════════════════════════

def _check_vector(v: Vector, name: str = "Vector") -> None:
    if not isinstance(v, list) or len(v) == 0:
        raise ValueError(f"{name} must be a non-empty list, got: {type(v)}")

def _check_rect(A: Matrix, name: str = "Matrix") -> None:
    """Verify A is a non-empty, non-jagged 2-D list."""
    if not A or not isinstance(A, list):
        raise ValueError(f"{name} must be a non-empty list of lists")
    if not A[0] or not isinstance(A[0], list):
        raise ValueError(f"{name} must have non-empty rows")
    ncols = len(A[0])
    for idx, row in enumerate(A):
        if len(row) != ncols:
            raise ValueError(
                f"{name} is not rectangular: "
                f"row 0 has {ncols} cols, row {idx} has {len(row)} cols"
            )

def shape(A: Matrix):
    """Return (rows, cols).  Assumes A is rectangular."""
    return (len(A), len(A[0]))


# ═══════════════════════════════════════════════════════════
# VECTOR OPERATIONS
# ═══════════════════════════════════════════════════════════

def dot(a: Vector, b: Vector) -> float:
    """
    Dot product of two vectors.

    Geometric meaning:
        dot(a, b) = ||a|| * ||b|| * cos(θ)
        - Zero  → a and b are orthogonal (perpendicular)
        - Max   → a and b point in the same direction (parallel)
        - Used in mat_mul to measure how much one direction 'projects onto' another

    Complexity: O(n)
    """
    _check_vector(a, "a")
    _check_vector(b, "b")
    if len(a) != len(b):
        raise ValueError(
            f"dot: vector length mismatch — len(a)={len(a)}, len(b)={len(b)}"
        )
    return sum(x * y for x, y in zip(a, b))


def vec_add(a: Vector, b: Vector) -> Vector:
    """Element-wise addition.  Complexity: O(n)"""
    _check_vector(a, "a")
    _check_vector(b, "b")
    if len(a) != len(b):
        raise ValueError(
            f"vec_add: length mismatch — len(a)={len(a)}, len(b)={len(b)}"
        )
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: Vector, b: Vector) -> Vector:
    """Element-wise subtraction.  Complexity: O(n)"""
    _check_vector(a, "a")
    _check_vector(b, "b")
    if len(a) != len(b):
        raise ValueError(
            f"vec_sub: length mismatch — len(a)={len(a)}, len(b)={len(b)}"
        )
    return [x - y for x, y in zip(a, b)]


def scalar_mul(v: Vector, s: float) -> Vector:
    """Scale every element of v by s.  Complexity: O(n)"""
    _check_vector(v, "v")
    return [x * s for x in v]


def vec_norm(v: Vector) -> float:
    """Euclidean (L2) norm of v.  Complexity: O(n)"""
    return math.sqrt(dot(v, v))


def vec_eq(a: Vector, b: Vector, tol: float = 1e-9) -> bool:
    """Element-wise approximate equality."""
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ═══════════════════════════════════════════════════════════
# MATRIX CONSTRUCTION HELPERS
# ═══════════════════════════════════════════════════════════

def zeros(rows: int, cols: int) -> Matrix:
    """Return an (rows × cols) zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    """Return the n×n identity matrix."""
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def mat_from_flat(flat: List[float], rows: int, cols: int) -> Matrix:
    """Reshape a flat list into a (rows × cols) matrix, row-major."""
    if len(flat) != rows * cols:
        raise ValueError(
            f"mat_from_flat: {len(flat)} elements cannot fill ({rows}×{cols})"
        )
    return [[flat[i * cols + j] for j in range(cols)] for i in range(rows)]


# ═══════════════════════════════════════════════════════════
# MATRIX OPERATIONS
# ═══════════════════════════════════════════════════════════

def transpose(A: Matrix) -> Matrix:
    """
    Return a fully independent transpose of A.

    Mutation of A will NOT corrupt the result (deep element copy).
    Shape: (m × n)  →  (n × m)
    Complexity: O(m * n)

    Why independence matters:
        If we returned a view/reference, downstream code could corrupt
        gradients during backprop by accidentally mutating original weights.
    """
    _check_rect(A, "A")
    m, n = shape(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]


def mat_add(A: Matrix, B: Matrix) -> Matrix:
    """Element-wise matrix addition.  Complexity: O(m * n)"""
    _check_rect(A, "A")
    _check_rect(B, "B")
    if shape(A) != shape(B):
        raise ValueError(
            f"mat_add: shape mismatch — {shape(A)} vs {shape(B)}"
        )
    m, n = shape(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]


def mat_scalar_mul(A: Matrix, s: float) -> Matrix:
    """Multiply every element of A by scalar s.  Complexity: O(m * n)"""
    _check_rect(A, "A")
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_vec_mul(A: Matrix, v: Vector) -> Vector:
    """
    Matrix-vector product:  A @ v
    Each element of the result is the dot product of a row of A with v.
    Shape: (m × n) @ (n,)  →  (m,)
    Complexity: O(m * n)
    """
    _check_rect(A, "A")
    _check_vector(v, "v")
    m, n = shape(A)
    if len(v) != n:
        raise ValueError(
            f"mat_vec_mul: shape mismatch — matrix ({m}×{n}), vector len={len(v)}"
        )
    return [dot(row, v) for row in A]


def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """
    Matrix multiplication:  C = A @ B

    C[i][j] = dot(row_i(A), col_j(B))
            = the composed linear transformation — apply B first, then A.

    Cache strategy:
        B is transposed once into Bt so that col_j(B) becomes row_j(Bt).
        Both loops then access contiguous memory (rows), minimising cache misses.

    Shape: (m × k) @ (k × n)  →  (m × n)
    Complexity: O(m * k * n)
    """
    _check_rect(A, "A")
    _check_rect(B, "B")
    m, k1 = shape(A)
    k2, n = shape(B)
    if k1 != k2:
        raise ValueError(
            f"mat_mul: inner dimension mismatch — "
            f"A is ({m}×{k1}), B is ({k2}×{n})"
        )
    Bt = transpose(B)          # Bt[j] = col j of B, now a contiguous row
    return [
        [dot(A[i], Bt[j]) for j in range(n)]
        for i in range(m)
    ]


def mat_hadamard(A: Matrix, B: Matrix) -> Matrix:
    """
    Element-wise (Hadamard) product.
    Used in backprop through activation functions.
    Shape: both must be (m × n).
    Complexity: O(m * n)
    """
    _check_rect(A, "A")
    _check_rect(B, "B")
    if shape(A) != shape(B):
        raise ValueError(
            f"mat_hadamard: shape mismatch — {shape(A)} vs {shape(B)}"
        )
    m, n = shape(A)
    return [[A[i][j] * B[i][j] for j in range(n)] for i in range(m)]


def mat_eq(A: Matrix, B: Matrix, tol: float = 1e-9) -> bool:
    """Element-wise approximate equality for two matrices."""
    if shape(A) != shape(B):
        return False
    m, n = shape(A)
    return all(
        abs(A[i][j] - B[i][j]) <= tol
        for i in range(m)
        for j in range(n)
    )


# ═══════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS  (scalar + vector forms)
# ═══════════════════════════════════════════════════════════

# ── ReLU ─────────────────────────────────────────────────

def relu(x: float) -> float:
    """
    Rectified Linear Unit — scalar form.
    relu(x) = max(0, x)

    Gradient: 1 if x > 0, 0 if x < 0, undefined (treated as 0) at x == 0.
    Key property: avoids vanishing gradients for positive inputs.
    """
    return max(0.0, x)

def relu_vec(v: Vector) -> Vector:
    return [relu(x) for x in v]

def relu_grad(x: float) -> float:
    """Derivative of relu at x (sub-gradient at 0 = 0)."""
    return 1.0 if x > 0.0 else 0.0

def relu_grad_vec(v: Vector) -> Vector:
    return [relu_grad(x) for x in v]


# ── Sigmoid ───────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """
    Logistic sigmoid — numerically stable form.

    Naïve:  1 / (1 + e^{-x})
    Problem: e^{-x} → ∞ for large negative x → OverflowError.

    Stable branches:
        x >= 0:  1 / (1 + e^{-x})     [e^{-x} is small → safe]
        x <  0:  e^x / (1 + e^x)      [e^x is small → safe]

    Both branches are algebraically identical (multiply ÷ by e^x),
    but each only evaluates a small exponential.

    Range: (0, 1) strictly — never exactly 0 or 1 in theory,
    though float64 saturates to 0.0 / 1.0 beyond ≈ ±710.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def sigmoid_vec(v: Vector) -> Vector:
    return [sigmoid(x) for x in v]

def sigmoid_grad(x: float) -> float:
    """
    Derivative of sigmoid:  σ(x) * (1 - σ(x))

    Note: saturates to ~0 for |x| >> 1.
    This is the root cause of vanishing gradients in deep sigmoid networks.
    """
    s = sigmoid(x)
    return s * (1.0 - s)

def sigmoid_grad_vec(v: Vector) -> Vector:
    return [sigmoid_grad(x) for x in v]


# ── Tanh ─────────────────────────────────────────────────

def tanh_scalar(x: float) -> float:
    """
    Hyperbolic tangent — derived from first principles.

    Definition: tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

    Numerically stable branches:
        x >= 0:  let u = e^{-2x}  →  (1 - u) / (1 + u)
                 [e^{-2x} is small for large positive x]
        x <  0:  let u = e^{2x}   →  (u - 1) / (u + 1)
                 [e^{2x} is small for large negative x]

    Range: (-1, 1).  Zero-centred — unlike sigmoid — which helps
    gradient flow in practice (activations don't all share same sign).

    Useful identity (not used here, but good to know):
        tanh(x) = 2 * sigmoid(2x) - 1
    """
    if x >= 0:
        u = math.exp(-2.0 * x)
        return (1.0 - u) / (1.0 + u)
    else:
        u = math.exp(2.0 * x)
        return (u - 1.0) / (u + 1.0)

def tanh_vec(v: Vector) -> Vector:
    return [tanh_scalar(x) for x in v]

def tanh_grad(x: float) -> float:
    """
    Derivative of tanh:  1 - tanh²(x)

    Also saturates (→ 0) for large |x|, but less aggressively than sigmoid.
    """
    t = tanh_scalar(x)
    return 1.0 - t * t

def tanh_grad_vec(v: Vector) -> Vector:
    return [tanh_grad(x) for x in v]