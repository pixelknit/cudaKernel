import numpy as np
import time
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

def mm(matrix1,matrix2):

    rows_matrix1, cols_matrix1 = matrix1.shape
    rows_matrix2, cols_matrix2 = matrix2.shape

    if cols_matrix1 != rows_matrix2:
        print("Error! incompatible matrix shapes")
        return

    result = np.zeros((rows_matrix1, cols_matrix2)).astype(np.float32)

    for i in range(rows_matrix1):
        for j in range(cols_matrix2):
            for k in range(cols_matrix1):
                result[i,j] += matrix1[i,k] * matrix2[k,j]


    return result

c_time = time.monotonic()
N = 256
A = np.random.rand(N,N).astype(np.float32)
B = np.random.rand(N,N).astype(np.float32)

C = mm(A,B)
e_time = time.monotonic()

print(C)
print("----------------")
print(f"{Fore.YELLOW}Time: {e_time - c_time}")