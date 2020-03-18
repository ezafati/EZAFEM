from typing import List, Tuple, Type
from GlobalObjects.MainObjects import Material, MatrixObj

import numpy as np


class MeshObj(object):

    def __init__(self, dim: int = 2, eltype: str = None, probtype: str = None, nbvertx: int = None):
        self.dim = dim
        self.eltype = eltype  # element type
        self.probtype = probtype  # problem type (plane strain  or stress if 2D)
        self.nbvertx = nbvertx
        self.parts = []

    def get_parts(self, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    if line.split(' ')[0] == 'PART_NAME':
                        self.parts.append(Part(label=line.split(' ')[1], dim=self.dim, probtype=self.probtype,
                                               eltype=self.eltype, nbvertx=self.nbvertx))
            except StopIteration:
                pass

    def get_part_plist(self, part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1] == part.label:
                            while True:
                                line = next(f)
                                if line.strip() == 'POINT_LIST':
                                    size = int(next(f).split(' ')[1])
                                    part.plist = np.ndarray(shape=(size, self.dim), dtype=np.float64)
                                    for npt in range(size):
                                        pt = next(f).split(' ')
                                        part.plist[npt, 0:2] = [float(pt[1]), float(pt[2])]
                                    break
                            break
                    except IndexError:
                        pass
            except StopIteration:
                pass

    def get_part_topology(self, part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1] == part.label:
                            while True:
                                line = next(f)
                                if line.strip() == 'TOPOLOGY':
                                    size = int(next(f).split(' ')[1])
                                    part.conn = np.ndarray(shape=(self.nbvertx, size), dtype=np.float64)
                                    for nel in range(size):
                                        el = next(f).split(' ')
                                        part.conn[:, nel] = [int(el[1]), int(el[2]), int(el[3])]
                                    break
                            break
                    except IndexError:
                        pass
            except StopIteration:
                pass

    def read_ezamesh(self, file):
        """Read  mesh file from EZAMESH
        and fill all the attributes of the instance"""
        self.dim = 2
        self.eltype = 'tri3'
        self.nbvertx = 3
        self.probtype = 'STRESS'
        self.get_parts(file)
        for part in self.parts:
            self.get_part_plist(part, file)
            self.get_part_topology(part, file)
            print(type(part.stiffmat))


class Part(MeshObj):
    stiffmat = MatrixObj(mtype='stiff')

    def __init__(self, label: str = 'PART', plist: List[Tuple[int]] = None, conn: 'Array' = None, gard: 'Array' = None,
                 mate: Type[Material] = None, dim: int = 2, eltype: str = None, probtype: str = None,
                 nbvertx: int = None):
        super().__init__(dim, eltype, probtype, nbvertx)
        self.label = label
        self.plist = plist
        self.conn = conn
        self.gard = gard
        self.mate = mate
        self.shape_grad = gard

    def grad_shape_array(self):
        eval(f'{self}.grad_shape_array_{self.eltype}')

    def grad_shape_array_tri3(self):
        """Compute the strain-displacement components and the
        determinant of the Jacobian matrix for each element in the
        mesh"""
        size_conn = self.conn.size[1]
        vct_tmp = np.zeros((4, size_conn), dtype=np.float64)
        for p in range(size_conn):
            el = self.conn[:, p]
            pa, pb, pc = [self.plist[p] for p in el]
            J = np.zeros((2, 2), dtype=np.float64)  # init jacob matrix
            J[0, 0] = -pa[1] + pb[1]
            J[0, 1] = pa[1] - pc[1]
            J[1, 0] = pa[0] - pb[0]
            J[1, 1] = -pa[0] + pc[0]
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

            Nav = J * np.array([-1, -1]).T / detJ  # Nav = (d(Na)/dx, d(Na)/dy)
            Nbv = J * np.array([1, 0]).T / detJ
            Ncv = J * np.array([0, 1]).T / detJ
            vct_tmp[:, p] = (Nav, Nbv, Ncv, detJ)
        self.shape_grad = vct_tmp
