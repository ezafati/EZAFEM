from typing import List, Tuple

import numpy as np

from globalvars import mesh


def initialize_deco(cls):
    cls.dim = mesh.dim
    cls.eltype = mesh.eltype
    cls.npts = len(mesh.plist)
    cls.nel = mesh.conn.size[1]
    cls.conn = mesh.conn
    return cls


class MeshObj:
    def __init__(self, label: str, dim: int, plist: List[Tuple[int]], conn: 'Array', eltype: str, gard: 'Array',  probtype: str):
        self.label = label
        self.dim = dim
        self.plist = plist
        self.conn = conn
        self.eltype = eltype
        self.shape_grad = gard
        self.probtype = probtype

    def grad_shape_array(self):
        eval(f'{self}.grad_shape_array_{self.eltype}')

    def grad_shape_array_tri3(self):
        size = self.conn.size[1]
        vct_tmp = np.zeros((4, size), dtype=np.float32)
        for p in range(size):
            el = self.conn[:, p]
            pa, pb, pc = [self.plist[p] for p in el]
            J = np.zeros((2, 2), dtype=np.float32)  # init jacob matrix
            J[0, 0] = -pa[1] + pb[1]
            J[0, 1] = pa[1] - pc[1]
            J[1, 0] = pa[0] - pb[0]
            J[1, 1] = -pa[0] + pc[0]
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

            Nav = J * np.array([-1, -1]).T / detJ
            Nbv = J * np.array([1, 0]).T / detJ
            Ncv = J * np.array([0, 1]).T / detJ
            vct_tmp[:, p] = (Nav, Nbv, Ncv, detJ)
        self.shape_grad = vct_tmp
