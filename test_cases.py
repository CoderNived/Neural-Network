# === mat_mul tests ===
# I @ A == A (identity)
I = [[1,0],[0,1]]
A = [[3,4],[5,6]]
# expect [[3,4],[5,6]]

# Shape: (2x3) @ (3x2) -> (2x2)
M1 = [[1,2,3],[4,5,6]]          # 2x3
M2 = [[7,8],[9,10],[11,12]]     # 3x2
# expect [[58,64],[139,154]]  -- computed by hand:
#   [0][0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
#   [0][1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64

# Shape mismatch: (3x2) @ (3x2) should raise ValueError

# === sigmoid tests ===
# sigmoid(0) = 1/(1+e^0) = 1/2 = 0.5  (by hand)
# sigmoid(100) -> should be ~1.0 (not overflow/NaN)
# sigmoid(-100) -> should be ~0.0

# === dot product ===
# dot([1,0],[0,1]) = 0  (orthogonal -> no projection)
# dot([1,0],[1,0]) = 1  (parallel -> full projection)
# dot([3,4],[3,4]) = 25 (= ||v||^2)