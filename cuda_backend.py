from numba import cuda
import numpy as np
import math
import time

# CUDA kernel
@cuda.jit
def matmul_kernel(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(bpg):
        sA[ty, tx] = A[y, tx + i * TPB]
        sB[ty, tx] = B[ty + i * TPB, x]

        cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        cuda.syncthreads()

    C[y, x] = tmp

N = 1024
TPB = 32

c_time = time.monotonic()

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(A.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(B.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

matmul_kernel[blockspergrid, threadsperblock](A, B, C)

e_time = time.monotonic()

print(C)
print("------------")
print("Time: ", e_time - c_time)