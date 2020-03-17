from typing import List, Tuple

import numpy as np


def initialize_deco(mesh):
    """ This function is used as decorator for classes to
    retrieve all the characteristics of the mesh """
    def decorator(cls):
        cls.dim = mesh.dim
        cls.eltype = mesh.eltype
        cls.conn = mesh.conn
        try:
            cls.npts = len(mesh.plist)
            cls.nel = mesh.conn.size[1]
            cls.nbvertx = mesh.conn.size[0]
        except Exception as e:
            cls.npts = None
            cls.nel = None
            cls.nbvertx = None
        return cls

    return decorator


class MeshObj(object):
    def __init__(self, label: str = 'MESH', dim: int = 2, plist: List[Tuple[int]] = None, conn: 'Array' = None,
                 eltype: str = None, gard: 'Array' = None, probtype: str = None, nbvertx: int = None):
        self.label = label
        self.dim = dim
        self.plist = plist  # points list
        self.conn = conn  # connectivity array
        self.eltype = eltype  # element type
        self.shape_grad = gard
        self.probtype = probtype  # problem type (plane strain  or stress if 2D)
        self.nbvertx = nbvertx

    def grad_shape_array(self):
        eval(f'{self}.grad_shape_array_{self.eltype}')

    def grad_shape_array_tri3(self):
        """Compute the strain-displacement components and the
        determinant of the Jacobian matrix for each element in the
        mesh"""
        size_conn = self.conn.size[1]
        vct_tmp = np.zeros((4, size_conn), dtype=np.float32)
        for p in range(size_conn):
            el = self.conn[:, p]
            pa, pb, pc = [self.plist[p] for p in el]
            J = np.zeros((2, 2), dtype=np.float32)  # init jacob matrix
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

    def read_ezamesh(self, file):
        self.dim = 2
        self.eltype = 'tri3'
        self.nbvertx = 3
        with open(file, 'r') as f:
            try:
                label = next(f).split(' ')[2]
                self.label = label
                f.seek(0,0)
                while True:
                    line = next(f)
                    if line.replace(' ', '').lower().strip() == 'pointlist':
                        size = int(next(f).split(' ')[1])
                        print(size)
                        self.plist = np.ndarray(shape=(size, self.dim), dtype=np.float64)
                        for npt in range(size):
                            pt = next(f).split(' ')
                            self.plist[npt, 0:2] = [float(pt[1]), float(pt[2])]
                        break
                f.seek(0, 0)
                while True:
                    line = next(f)
                    if line.lower().strip() == 'topology':
                        size = int(next(f).split(' ')[1])
                        self.conn = np.ndarray(shape=(3, size))
                        for nel in range(size):
                            el = next(f).split(' ')
                            self.conn[:, nel] = [int(el[1]), int(el[2]), int(el[3])]
                        break
            except StopIteration:
                print('END OF THE FILE IS REACHED')
