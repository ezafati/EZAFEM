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

res = np.array([1,2,3])
print(res.dot(res))
#print(arr)
#print(mat.shape, arr.shape)
#print(-arr.T*arr)
#print((arr*arr.T)[0,0])
#print(type(arr + np.zeros(5)))
#print(res.shape)
#print(lil_matrix(np.zeros((5,))).shape)
#print(norm(arr, ord=np.inf))
#print(type(np.inf))

