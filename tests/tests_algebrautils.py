from GlobalObjects.LinAlgebraUtils import *
import numpy as np

mat = np.array([[5, 2], [2, 3]], dtype=np.float64)
print(mat)

mat = csr_matrix(mat)
print(f'before preconditioning the matrice mat is \n {mat}')

M = precond_ic(mat)
print(f'after preconditioning the matrice  M  is\n {M}')

Mm = M * mat
print('product M*mat is \n ', Mm)

mat2 = lil_matrix((3, 3))
mat2[1, 1] = 2
mat2[0, 0] = 1
mat2[2, 2] = 5
mat2[0, 2] = 5
mat2[2, 0] = 5

mat2 = csr_matrix(mat2)
x = np.empty((3, 1))
b = np.random.random((3, 1))
res, count = cg_method(x, mat2, b)

print(b, res, count)
