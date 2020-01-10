from typing import Type

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm


def _assembly_mass_tri3(mat, density):
    return 'not implemented'


def cg_method(x: 'Array', mat: Type[csr_matrix], b: 'Array', **kwargs):
    """Conjugate gradient method without preconditioning"""
    if 'eps' in kwargs:
        eps = kwargs['eps']
    else:
        eps = 1e-5
    r = mat * x - b
    try:
        ratio = norm(r, ord=np.inf) / norm(b, ord=np.inf)
    except ZeroDivisionError:
        size = x.shape[0]
        x = np.zeros(size)
        return x
    p = -r
    count = 0
    while ratio < eps and count < 100:
        fact = mat * p
        alpha = r.dot(r)/ p.dot(fact)
        x = x + alpha * p
        rp = r + alpha * fact
        beta = rp.dot(rp)[0] / r.dot(r)
        p = -rp + beta * p
        r = rp
        count += 1
    return x


def precond_ic(mat: Type[csr_matrix]):
    """return preconditionner using incomplete Cholesky decomposition"""
    pass
