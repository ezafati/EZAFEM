import numpy as np
from numpy import transpose

from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse.linalg import norm

row = np.array([1, 1, 1, 4, 2, 4, 3])
col = np.array([0, 2, 4, 0, 4, 3, 1])

data = np.array([0.1, 0.2, 1.5, 1, 0.4, 0.3, 0.1])

mat = csr_matrix((data, (row, col)), shape=(5, 5))

matcoo = coo_matrix(mat)
print(mat.indptr)
print(mat.indices)

matt = mat.copy()
print(matt.shape[0])
arr = lil_matrix(np.zeros(5))
arr[0, 4] = 1

res = np.array([1, 2, 3])
# print(res.dot(res))


matr = lil_matrix((3,1))
matr[1] = 1
print(matr)
