from GlobalObjects.LinAlgebraUtils import *
import numpy as np

mat = np.array([[10, 2, 3], [2, 20, 4], [3, 4, 50]])
print(mat)

mat = csr_matrix(mat)
print(f'before preconditioning the matrice is \n {mat}')

M = precond_ic(mat)
print(f'after preconditioning the matrice is \n {M}')

Mm = M*mat

print('product M*mat is \n ', Mm)
