import numpy as np
import random
import time
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

def mm(matrix1,matrix2):
    result = []
    for i, row in enumerate(matrix1):
        newRow = []
        for j, row in enumerate(row):
            dot_product = sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2)))
            newRow.append(dot_product)
        result.append(newRow)

    return result

def create_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]


matrix1 = np.array([[2,4,5,1],
                    [2,1,2,2],
                    [8,1,2,3],
                    [1,3,2,8]
                    ])

matrix2 = np.array([[2,4,5,1],
                    [2,1,2,2],
                    [8,1,2,3],
                    [1,3,2,8]
                    ])

c_time = time.monotonic()
N = 1024
#A = np.random.rand(N,N).astype(np.float32)
#B = np.random.rand(N,N).astype(np.float32)

a = create_matrix(N)
b = create_matrix(N)

C = mm(a,b)
print(C)
e_time = time.monotonic()
print("----------------")
print(f"{Fore.YELLOW}Time: {e_time - c_time}")
