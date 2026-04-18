"""
tests/test_linalg.py
--------------------
Exhaustive test suite for engine/linalg.py.

Run:  python tests/test_linalg.py
      (or: python -m pytest tests/test_linalg.py -v)

Structure:
    TestVectors          — dot, vec_add, vec_sub, scalar_mul, vec_norm
    TestMatrixHelpers    — shape, zeros, identity, mat_from_flat
    TestTranspose        — correctness + independence guarantee
    TestMatMul           — identity, known values, associativity, shape errors
    TestMatVecMul        — identity, known values, shape errors
    TestMatOps           — mat_add, mat_scalar_mul, mat_hadamard
    TestActivations      — relu, sigmoid, tanh (values, gradients, stability)
    TestEdgeCases        — empty inputs, non-rectangular matrices, type guards
"""

import math
import sys
import os
import unittest

# Allow running from repo root or from tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.linalg import (
    # vectors
    dot, vec_add, vec_sub, scalar_mul, vec_norm, vec_eq,
    # matrix helpers
    shape, zeros, identity, mat_from_flat,
    # matrix ops
    transpose, mat_add, mat_scalar_mul, mat_vec_mul,
    mat_mul, mat_hadamard, mat_eq,
    # activations
    relu, relu_vec, relu_grad, relu_grad_vec,
    sigmoid, sigmoid_vec, sigmoid_grad, sigmoid_grad_vec,
    tanh_scalar, tanh_vec, tanh_grad, tanh_grad_vec,
)

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ═══════════════════════════════════════════════════════════
# VECTOR TESTS
# ═══════════════════════════════════════════════════════════

class TestVectors(unittest.TestCase):

    # ── dot ───────────────────────────────────────────────

    def test_dot_orthogonal(self):
        """Orthogonal vectors → 0."""
        self.assertEqual(dot([1, 0], [0, 1]), 0)

    def test_dot_parallel(self):
        """Parallel unit vectors → 1."""
        self.assertEqual(dot([1, 0], [1, 0]), 1)

    def test_dot_norm_squared(self):
        """dot(v, v) == ||v||²"""
        self.assertEqual(dot([3, 4], [3, 4]), 25)

    def test_dot_negative(self):
        """Anti-parallel vectors → negative."""
        self.assertEqual(dot([1, 0], [-1, 0]), -1)

    def test_dot_multivariate(self):
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32)

    def test_dot_length_mismatch(self):
        with self.assertRaises(ValueError):
            dot([1, 2], [1])

    def test_dot_empty_raises(self):
        with self.assertRaises((ValueError, IndexError)):
            dot([], [])

    # ── vec_add ───────────────────────────────────────────

    def test_vec_add_basic(self):
        self.assertEqual(vec_add([1, 2, 3], [4, 5, 6]), [5, 7, 9])

    def test_vec_add_negative(self):
        self.assertEqual(vec_add([1, -1], [-1, 1]), [0, 0])

    def test_vec_add_mismatch(self):
        with self.assertRaises(ValueError):
            vec_add([1, 2], [1])

    # ── vec_sub ───────────────────────────────────────────

    def test_vec_sub_basic(self):
        self.assertEqual(vec_sub([5, 7, 9], [4, 5, 6]), [1, 2, 3])

    def test_vec_sub_self_is_zero(self):
        v = [3.0, 4.0]
        self.assertTrue(vec_eq(vec_sub(v, v), [0.0, 0.0]))

    # ── scalar_mul ────────────────────────────────────────

    def test_scalar_mul_positive(self):
        self.assertEqual(scalar_mul([1, 2, 3], 2), [2, 4, 6])

    def test_scalar_mul_zero(self):
        self.assertEqual(scalar_mul([1, 2, 3], 0), [0, 0, 0])

    def test_scalar_mul_negative(self):
        self.assertEqual(scalar_mul([1, -2], -1), [-1, 2])

    # ── vec_norm ──────────────────────────────────────────

    def test_norm_3_4(self):
        self.assertAlmostEqual(vec_norm([3.0, 4.0]), 5.0, places=12)

    def test_norm_unit_vector(self):
        self.assertAlmostEqual(vec_norm([1.0, 0.0, 0.0]), 1.0, places=12)

    def test_norm_zero_vector(self):
        self.assertAlmostEqual(vec_norm([0.0, 0.0, 0.0]), 0.0, places=12)


# ═══════════════════════════════════════════════════════════
# MATRIX HELPER TESTS
# ═══════════════════════════════════════════════════════════

class TestMatrixHelpers(unittest.TestCase):

    def test_shape_basic(self):
        self.assertEqual(shape([[1, 2, 3], [4, 5, 6]]), (2, 3))

    def test_shape_square(self):
        self.assertEqual(shape([[1, 2], [3, 4]]), (2, 2))

    def test_zeros(self):
        Z = zeros(2, 3)
        self.assertEqual(shape(Z), (2, 3))
        self.assertTrue(all(Z[i][j] == 0.0 for i in range(2) for j in range(3)))

    def test_identity_2x2(self):
        I = identity(2)
        self.assertEqual(I, [[1.0, 0.0], [0.0, 1.0]])

    def test_identity_3x3(self):
        I = identity(3)
        for i in range(3):
            for j in range(3):
                self.assertEqual(I[i][j], 1.0 if i == j else 0.0)

    def test_mat_from_flat_basic(self):
        M = mat_from_flat([1, 2, 3, 4, 5, 6], 2, 3)
        self.assertEqual(M, [[1, 2, 3], [4, 5, 6]])

    def test_mat_from_flat_wrong_size(self):
        with self.assertRaises(ValueError):
            mat_from_flat([1, 2, 3], 2, 3)


# ═══════════════════════════════════════════════════════════
# TRANSPOSE TESTS
# ═══════════════════════════════════════════════════════════

class TestTranspose(unittest.TestCase):

    def test_shape_2x3_becomes_3x2(self):
        A = [[1, 2, 3], [4, 5, 6]]
        At = transpose(A)
        self.assertEqual(shape(At), (3, 2))

    def test_values_correct(self):
        A = [[1, 2, 3], [4, 5, 6]]
        At = transpose(A)
        self.assertEqual(At[0], [1, 4])
        self.assertEqual(At[1], [2, 5])
        self.assertEqual(At[2], [3, 6])

    def test_double_transpose_is_original(self):
        A = [[1, 2], [3, 4], [5, 6]]
        self.assertTrue(mat_eq(transpose(transpose(A)), A))

    def test_independence_row_mutation(self):
        """Mutating A must NOT affect transpose(A)."""
        A = [[1, 2, 3], [4, 5, 6]]
        At = transpose(A)
        A[0][0] = 999
        self.assertEqual(At[0][0], 1,
            "transpose result was corrupted by mutation of original")

    def test_independence_transpose_mutation(self):
        """Mutating transpose(A) must NOT affect A."""
        A = [[1, 2], [3, 4]]
        At = transpose(A)
        At[0][0] = 999
        self.assertEqual(A[0][0], 1,
            "original matrix was corrupted by mutation of transpose")

    def test_square_symmetric(self):
        """Symmetric matrix: Aᵀ == A."""
        A = [[1, 2], [2, 4]]
        self.assertTrue(mat_eq(transpose(A), A))


# ═══════════════════════════════════════════════════════════
# MAT_MUL TESTS
# ═══════════════════════════════════════════════════════════

class TestMatMul(unittest.TestCase):

    def test_identity_left(self):
        I = identity(2)
        M = [[3, 4], [5, 6]]
        self.assertTrue(mat_eq(mat_mul(I, M), M), "I @ M must equal M")

    def test_identity_right(self):
        I = identity(2)
        M = [[3, 4], [5, 6]]
        self.assertTrue(mat_eq(mat_mul(M, I), M), "M @ I must equal M")

    def test_2x3_times_3x2(self):
        """
        [[1,2,3],   @  [[7, 8],    =  [[58,  64],
         [4,5,6]]       [9,10],        [139,154]]
                        [11,12]]
        Verify by hand:
            C[0][0] = 1*7 + 2*9 + 3*11 = 58
            C[0][1] = 1*8 + 2*10 + 3*12 = 64
        """
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        C = mat_mul(A, B)
        self.assertEqual(shape(C), (2, 2))
        self.assertEqual(C[0][0], 58)
        self.assertEqual(C[0][1], 64)
        self.assertEqual(C[1][0], 139)
        self.assertEqual(C[1][1], 154)

    def test_result_shape(self):
        """(m×k) @ (k×n) → (m×n)"""
        A = zeros(4, 3)
        B = zeros(3, 5)
        C = mat_mul(A, B)
        self.assertEqual(shape(C), (4, 5))

    def test_associativity(self):
        """(AB)C == A(BC) — must hold to float precision."""
        A = [[1, 2], [3, 4]]
        B = [[0, 1], [1, 0]]
        C = [[2, 0], [0, 2]]
        lhs = mat_mul(mat_mul(A, B), C)
        rhs = mat_mul(A, mat_mul(B, C))
        self.assertTrue(mat_eq(lhs, rhs), f"Associativity failed:\nlhs={lhs}\nrhs={rhs}")

    def test_non_commutativity(self):
        """AB ≠ BA in general."""
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        AB = mat_mul(A, B)
        BA = mat_mul(B, A)
        self.assertFalse(mat_eq(AB, BA), "AB == BA should be False for these matrices")

    def test_shape_mismatch_raises(self):
        """(3×2) @ (3×2) — inner dims 2 ≠ 3 → ValueError."""
        A = [[1, 2], [3, 4], [5, 6]]  # 3×2
        B = [[1, 2], [3, 4], [5, 6]]  # 3×2
        with self.assertRaises(ValueError):
            mat_mul(A, B)

    def test_non_square_identity(self):
        """Multiplying by identity for non-square matrix."""
        A = [[1, 2, 3], [4, 5, 6]]   # 2×3
        I3 = identity(3)
        self.assertTrue(mat_eq(mat_mul(A, I3), A))

    def test_zero_matrix(self):
        """A @ 0 == 0 for all A."""
        A = [[1, 2], [3, 4]]
        Z = zeros(2, 2)
        self.assertTrue(mat_eq(mat_mul(A, Z), Z))


# ═══════════════════════════════════════════════════════════
# MAT_VEC_MUL TESTS
# ═══════════════════════════════════════════════════════════

class TestMatVecMul(unittest.TestCase):

    def test_identity(self):
        I = identity(2)
        self.assertEqual(mat_vec_mul(I, [3.0, 5.0]), [3.0, 5.0])

    def test_scaling(self):
        """Diagonal matrix scales each component."""
        D = [[2.0, 0.0], [0.0, 3.0]]
        self.assertEqual(mat_vec_mul(D, [4.0, 5.0]), [8.0, 15.0])

    def test_known_values(self):
        A = [[1, 2, 3], [4, 5, 6]]
        v = [1, 0, -1]
        result = mat_vec_mul(A, v)
        self.assertEqual(result, [1*1 + 2*0 + 3*(-1), 4*1 + 5*0 + 6*(-1)])
        self.assertEqual(result, [-2, -2])

    def test_shape_mismatch(self):
        A = [[1, 2], [3, 4]]
        with self.assertRaises(ValueError):
            mat_vec_mul(A, [1, 2, 3])


# ═══════════════════════════════════════════════════════════
# MAT_ADD / MAT_SCALAR_MUL / MAT_HADAMARD
# ═══════════════════════════════════════════════════════════

class TestMatOps(unittest.TestCase):

    def test_mat_add_basic(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        self.assertTrue(mat_eq(mat_add(A, B), [[6, 8], [10, 12]]))

    def test_mat_add_zero(self):
        A = [[1, 2], [3, 4]]
        Z = zeros(2, 2)
        self.assertTrue(mat_eq(mat_add(A, Z), A))

    def test_mat_add_mismatch(self):
        with self.assertRaises(ValueError):
            mat_add([[1, 2]], [[1, 2], [3, 4]])

    def test_mat_scalar_mul(self):
        A = [[1, 2], [3, 4]]
        self.assertTrue(mat_eq(mat_scalar_mul(A, 2.0), [[2, 4], [6, 8]]))

    def test_mat_scalar_mul_zero(self):
        A = [[1, 2], [3, 4]]
        self.assertTrue(mat_eq(mat_scalar_mul(A, 0.0), zeros(2, 2)))

    def test_hadamard_basic(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        expected = [[5, 12], [21, 32]]
        self.assertTrue(mat_eq(mat_hadamard(A, B), expected))

    def test_hadamard_with_zeros(self):
        A = [[1, 2], [3, 4]]
        Z = zeros(2, 2)
        self.assertTrue(mat_eq(mat_hadamard(A, Z), Z))

    def test_hadamard_mismatch(self):
        with self.assertRaises(ValueError):
            mat_hadamard([[1, 2]], [[1, 2], [3, 4]])


# ═══════════════════════════════════════════════════════════
# ACTIVATION FUNCTION TESTS
# ═══════════════════════════════════════════════════════════

class TestReLU(unittest.TestCase):

    def test_positive(self):
        self.assertEqual(relu(3.0), 3.0)

    def test_negative(self):
        self.assertEqual(relu(-3.0), 0.0)

    def test_zero(self):
        self.assertEqual(relu(0.0), 0.0)

    def test_vec(self):
        self.assertEqual(relu_vec([1.0, -1.0, 0.0]), [1.0, 0.0, 0.0])

    def test_grad_positive(self):
        self.assertEqual(relu_grad(3.0), 1.0)

    def test_grad_negative(self):
        self.assertEqual(relu_grad(-3.0), 0.0)

    def test_grad_zero(self):
        """Sub-gradient convention: relu_grad(0) == 0."""
        self.assertEqual(relu_grad(0.0), 0.0)

    def test_grad_vec(self):
        self.assertEqual(relu_grad_vec([2.0, -1.0, 0.0]), [1.0, 0.0, 0.0])


class TestSigmoid(unittest.TestCase):

    def test_at_zero(self):
        """σ(0) = 0.5 — by definition."""
        self.assertAlmostEqual(sigmoid(0), 0.5, places=15)

    def test_large_positive(self):
        """σ(100) ≈ 1.0"""
        self.assertAlmostEqual(sigmoid(100), 1.0, places=6)

    def test_large_negative(self):
        """σ(-100) ≈ 0.0"""
        self.assertAlmostEqual(sigmoid(-100), 0.0, places=6)

    def test_stability_large_positive(self):
        """Must not raise OverflowError for very large positive input."""
        try:
            result = sigmoid(1000)
        except OverflowError:
            self.fail("sigmoid(1000) raised OverflowError")
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_stability_large_negative(self):
        """Must not raise OverflowError for very large negative input."""
        try:
            result = sigmoid(-1000)
        except OverflowError:
            self.fail("sigmoid(-1000) raised OverflowError")
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_symmetry(self):
        """σ(x) + σ(-x) == 1 for all x."""
        for x in [0.1, 1.0, 5.0, 10.0, 100.0]:
            self.assertAlmostEqual(sigmoid(x) + sigmoid(-x), 1.0, places=12,
                msg=f"Symmetry failed at x={x}")

    def test_monotone_increasing(self):
        """σ must be strictly increasing."""
        xs = [-10.0, -1.0, 0.0, 1.0, 10.0]
        vals = [sigmoid(x) for x in xs]
        for i in range(len(vals) - 1):
            self.assertLess(vals[i], vals[i + 1],
                msg=f"sigmoid not increasing at x={xs[i]}")

    def test_vec(self):
        result = sigmoid_vec([0.0, 100.0, -100.0])
        self.assertAlmostEqual(result[0], 0.5, places=15)
        self.assertAlmostEqual(result[1], 1.0, places=6)
        self.assertAlmostEqual(result[2], 0.0, places=6)

    def test_grad_at_zero(self):
        """σ'(0) = σ(0) * (1 - σ(0)) = 0.25"""
        self.assertAlmostEqual(sigmoid_grad(0), 0.25, places=15)

    def test_grad_saturation(self):
        """
        Gradient saturates near 0 for large |x|.
        This is the vanishing gradient problem for sigmoid.
        """
        self.assertAlmostEqual(sigmoid_grad(10.0), 0.0, places=4)
        self.assertAlmostEqual(sigmoid_grad(-10.0), 0.0, places=4)

    def test_grad_nonnegative(self):
        """Sigmoid gradient is always ≥ 0 (monotone function)."""
        for x in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            self.assertGreaterEqual(sigmoid_grad(x), 0.0)


class TestTanh(unittest.TestCase):

    def test_at_zero(self):
        """tanh(0) = 0"""
        self.assertAlmostEqual(tanh_scalar(0), 0.0, places=15)

    def test_large_positive(self):
        """tanh(100) ≈ 1.0"""
        self.assertAlmostEqual(tanh_scalar(100), 1.0, places=6)

    def test_large_negative(self):
        """tanh(-100) ≈ -1.0"""
        self.assertAlmostEqual(tanh_scalar(-100), -1.0, places=6)

    def test_matches_math_tanh(self):
        """Must agree with math.tanh to full float precision."""
        for x in [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0]:
            self.assertAlmostEqual(tanh_scalar(x), math.tanh(x), places=12,
                msg=f"tanh mismatch at x={x}")

    def test_odd_function(self):
        """tanh(-x) == -tanh(x)"""
        for x in [0.1, 1.0, 5.0]:
            self.assertAlmostEqual(tanh_scalar(-x), -tanh_scalar(x), places=12,
                msg=f"Odd function failed at x={x}")

    def test_stability(self):
        """Must not raise for extreme inputs."""
        try:
            tanh_scalar(1000)
            tanh_scalar(-1000)
        except OverflowError:
            self.fail("tanh raised OverflowError on extreme input")

    def test_vec(self):
        result = tanh_vec([0.0, 100.0, -100.0])
        self.assertAlmostEqual(result[0], 0.0, places=15)
        self.assertAlmostEqual(result[1], 1.0, places=6)
        self.assertAlmostEqual(result[2], -1.0, places=6)

    def test_grad_at_zero(self):
        """tanh'(0) = 1 - tanh²(0) = 1"""
        self.assertAlmostEqual(tanh_grad(0.0), 1.0, places=15)

    def test_grad_saturation(self):
        """tanh gradient → 0 for large |x|."""
        self.assertAlmostEqual(tanh_grad(10.0), 0.0, places=4)
        self.assertAlmostEqual(tanh_grad(-10.0), 0.0, places=4)

    def test_grad_nonnegative(self):
        """tanh gradient is always ≥ 0."""
        for x in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            self.assertGreaterEqual(tanh_grad(x), 0.0)


# ═══════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_empty_matrix_raises(self):
        with self.assertRaises(ValueError):
            mat_mul([], [[1, 2]])

    def test_empty_row_raises(self):
        with self.assertRaises(ValueError):
            mat_mul([[]], [[1, 2]])

    def test_jagged_matrix_raises(self):
        """Non-rectangular matrix must be caught."""
        with self.assertRaises(ValueError):
            mat_mul([[1, 2], [3]], [[1], [2]])

    def test_mat_mul_1x1(self):
        """1×1 matrix multiplication = scalar multiplication."""
        A = [[3.0]]
        B = [[4.0]]
        self.assertTrue(mat_eq(mat_mul(A, B), [[12.0]]))

    def test_transpose_1d_row(self):
        """Transpose of (1×n) is (n×1)."""
        A = [[1, 2, 3]]
        At = transpose(A)
        self.assertEqual(shape(At), (3, 1))
        self.assertEqual(At, [[1], [2], [3]])

    def test_dot_single_element(self):
        self.assertEqual(dot([5.0], [3.0]), 15.0)

    def test_sigmoid_output_range(self):
        """Sigmoid output must always be in (0, 1)."""
        for x in [-1000.0, -1.0, 0.0, 1.0, 1000.0]:
            s = sigmoid(x)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_tanh_output_range(self):
        """tanh output must always be in [-1, 1]."""
        for x in [-1000.0, -1.0, 0.0, 1.0, 1000.0]:
            t = tanh_scalar(x)
            self.assertGreaterEqual(t, -1.0)
            self.assertLessEqual(t, 1.0)

    def test_relu_large_negative(self):
        """ReLU clamps all negative values to 0, including large ones."""
        self.assertEqual(relu(-1e10), 0.0)

    def test_mat_mul_with_identity_3x3(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        I = identity(3)
        self.assertTrue(mat_eq(mat_mul(A, I), A))
        self.assertTrue(mat_eq(mat_mul(I, A), A))


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)