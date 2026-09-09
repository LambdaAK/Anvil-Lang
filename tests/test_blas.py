"""Matrix products through Accelerate (cblas_sgemm) must agree with the interpreter."""
import pytest

from util import check_native, compile_text

from anvil.backend.aarch64 import generate


def gemm_calls(src):
    asm, _ = generate(compile_text(src).program)
    return asm.count("bl _cblas_sgemm")


@pytest.mark.parametrize("a_t, b_t", [(False, False), (True, False), (False, True), (True, True)])
def test_transposes(a_t, b_t):
    a = "A[k, i]" if a_t else "A[i, k]"
    b = "B[j, k]" if b_t else "B[k, j]"
    src = f"""
A: {"[40, 50]" if a_t else "[50, 40]"} ~ normal(0, 1)
B: {"[30, 40]" if b_t else "[40, 30]"} ~ normal(0, 1)
C[i, j] = sum {a} * {b}
print(C)
"""
    check_native(src)
    assert gemm_calls(src) == 1


def test_epilogues_accumulation_and_slices():
    src = """
x: [64, 100] ~ normal(0, 1)
W: [100, 48] ~ normal(0, 0.1)
b: [48] ~ normal(0, 1)
h = relu(x @ W + b)                    # the product, then an epilogue kernel
acc: [64, 48] = 1
acc[i, j] += sum x[i, k] * W[k, j]     # beta = 1
top = x[10:42] @ W                     # an operand that starts at an offset
print(sum(h), sum(acc), sum(top), h[3, 0:5])
"""
    check_native(src)
    assert gemm_calls(src) == 3


def test_overlap_and_small_products_stay_native():
    src = """
S: [8, 8] ~ normal(0, 1)
T = S @ S                              # too small: Anvil's own kernel
M: [64, 64] ~ normal(0, 0.1)
for step in range(3):
    M = M @ M + 0.01                   # the result replaces an operand: split, then the epilogue
print(sum(T), sum(M))
"""
    check_native(src)
    assert gemm_calls(src) == 1


def test_training_gradients_through_blas():
    src = """
x: [128, 64] ~ normal(0, 1)
y: i32[128] ~ randint(0, 10)
param W1: [64, 96] ~ normal(0, 0.1)
param W2: [96, 10] ~ normal(0, 0.1)
for step in range(5):
    loss = cross_entropy(relu(x @ W1) @ W2, y)
    minimize loss with adam(lr=1e-2)
    print(loss)
"""
    check_native(src)
    assert gemm_calls(src) >= 3           # the large forward and backward products


def test_batched_and_summed_products():
    src = """
q: [3, 128, 2, 64] ~ normal(0, 1)
k: [3, 128, 2, 64] ~ normal(0, 1)
s[b, h, i, j] = sum q[b, i, h, e] * k[b, j, h, e]          # a stack of 3·2 products (heads)
W: [64, 64] ~ normal(0, 1)
x: [4, 256, 64] ~ normal(0, 1)
y[n, i, o] = sum x[n, i, c] * W[c, o] + 1.0                  # W shared by the stack, then an epilogue
g[c, o] = sum x[n, i, c] * y[n, i, o]                        # (n, i) collapse: one product of depth 1024
xt: [4, 64, 256] ~ normal(0, 1)
h[c, o] = sum xt[n, c, i] * y[n, i, o]                       # a sum of 4 products (n and i do not collapse)
print(sum(s), s[2, 1, 5, 0:4], sum(y), g[3, 0:4], h[5, 0:3])
"""
    check_native(src)
    asm, _ = generate(compile_text(src).program)
    calls = [line for line in asm.splitlines() if "bl _cblas_sgemm" in line]
    assert len(calls) == 4, calls
    assert sum("× 6 on" in c for c in calls) == 1 and sum("× 4 on" in c for c in calls) == 1, calls   # y: one product
