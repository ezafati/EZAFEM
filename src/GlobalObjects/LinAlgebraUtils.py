from math import sqrt
from typing import Type

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def cg_method(x: 'Array', mat: Type[csr_matrix], b: 'Array', **kwargs):
    """Conjugate gradient method without preconditioning"""
    if 'eps' in kwargs:
        eps = kwargs['eps']
    else:
        eps = 1e-5
    r = mat * x - b
    try:
        ratio = np.linalg.norm(r, ord=np.inf) / np.linalg.norm(b, ord=np.inf)
    except ZeroDivisionError:
        size = x.shape[0]
        x = np.zeros(size)
        return x
    p = -r
    count = 0
    while ratio > eps and count < 100:
        fact = mat * p
        alpha = r.T.dot(r) / p.T.dot(fact)
        x = x + alpha * p
        rp = r + alpha * fact
        beta = rp.T.dot(rp) / r.T.dot(r)
        p = -rp + beta * p
        r = rp
        ratio = np.linalg.norm(r, ord=np.inf) / np.linalg.norm(b, ord=np.inf)
        count += 1
    return x, count


def precond_ic(mat: Type[csr_matrix]):
    """return preconditionner using incomplete
    Cholesky decomposition"""
    M = mat.copy()
    indices = M.indices
    indptr = M.indptr
    dim = M.shape[0]
    for i in range(1, dim):
        count = 0
        for k in indices[indptr[i]:indptr[i + 1]]:
            if k < i:
                count += 1
                M[i, k] = M[i, k] / M[k, k]
                for j in indices[indptr[i] + count:indptr[i + 1]]:
                    M[i, j] = M[i, j] - M[i, k] * M[k, j]
    Mu_inv = lil_matrix(M.shape, dtype=np.float64)
    for i in range(dim):
        Mu_inv[i, i] = 1 / sqrt(M[i, i])
    for col in range(1, dim):
        rowlist = reversed(range(col))
        for row in rowlist:
            for j in indices[indptr[row]:indptr[row + 1]]:
                if j > row:
                    Mu_inv[row, col] -= M[row, j] * Mu_inv[j, col]
            Mu_inv[row, col] /= M[row, row]
    Ml_inv = Mu_inv.T
    return csr_matrix(Mu_inv * Ml_inv)
