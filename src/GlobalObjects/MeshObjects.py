from typing import List, Tuple, Type
from GlobalObjects.MatrixObjects import MatrixObj, _mat_assembly_2d, VectObject
from GlobalObjects.MathUtils import GaussPoints
from GlobalObjects.MaterialObjects import Material
import numpy as np
import os.path
import yaml

MAIN_YAML_PATH = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'main.yml')
MATERIAL_DB = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'matdb/matdb.json')
GAUSS_POINTS_JSON = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'matdb/gauss.json')


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
                        self.parts.append(Part(label=line.split(' ')[1].strip(), dim=self.dim, probtype=self.probtype,
                                               eltype=self.eltype, nbvertx=self.nbvertx))
            except StopIteration:
                pass

    def get_part_plist(self, part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1].strip() == part.label:
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
                        if line.split(' ')[1].strip() == part.label:
                            while True:
                                line = next(f)
                                if line.strip() == 'TOPOLOGY':
                                    size = int(next(f).split(' ')[1])
                                    part.conn = np.ndarray(shape=(self.nbvertx, size), dtype=np.int)
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
            part.grad_shape_array()
            part.get_part_material()
            part.initiliaze_eps_array()


class Part(MeshObj):
    stiffmat = MatrixObj(mtype='stiff')
    massmat = MatrixObj(mtype='mass')
    dispvct = VectObject(vtype='disp')
    forcvct = VectObject(vtype='forc')

    def __init__(self, label: str = 'PART', plist: List[Tuple[int]] = None, conn: 'Array' = None, gard: 'Array' = None,
                 mate: Type[Material] = None, dim: int = 2, eltype: str = None, probtype: str = None,
                 nbvertx: int = None, defoarray: 'Array' = None):
        super().__init__(dim, eltype, probtype, nbvertx)
        self.label = label
        self.plist = plist
        self.conn = conn
        self.gard = gard
        self.mate = mate
        self.shape_grad = gard
        self.defo_array = defoarray
        self.gauss_points = None
        self.eps_array = None

    def __repr__(self):
        return f'{self.__class__.__name__}(label={self.label}, eltype={self.eltype}, mate={self.mate}, ' \
               f'probtype={self.probtype}, dim={self.dim})'

    def initiliaze_eps_array(self):
        ngp = len(self.gauss_points)
        nel = self.conn.shape[1]
        self.eps_array = np.ndarray((ngp, 3, nel), dtype=np.float64)

    def get_gauss_points(self):
        self.gauss_points = GaussPoints(eltype=self.eltype)
        self.gauss_points.get_gauss_points(GAUSS_POINTS_JSON)

    def get_part_material(self):
        with open(MAIN_YAML_PATH, 'r') as f:
            d = yaml.load(f.read(), Loader=yaml.Loader)
        try:
            mat_list = d['materials']
            for p in mat_list:
                if p['part'] == self.label:
                    self.mate = Material()
                    self.mate.name = p['type']
                    self.mate.get_material(MATERIAL_DB, self.conn.shape[1], len(self.gauss_points))
        except KeyError:
            print("DEFAULT MATERIAL WILL BE ASSIGNED TO THE PARTS")
            self.mate = Material(name='Linear_Elastic')
            self.mate.get_material(MATERIAL_DB, self.conn.shape[1])

    def mat_assembly(self, mtype, **kwargs):
        """assembly matrix according to the
        matrix type"""
        mat = eval(f'self.{mtype}mat')
        eval(f'_mat_assembly_{self.dim}d(self, mtype, **kwargs)')

    def grad_shape_array(self):
        """"compute the gradient shape array
        """
        eval(f'self.grad_shape_array_{self.eltype}()')

    def grad_shape_array_tri3(self):
        """Compute the strain-displacement components and the
        determinant of the Jacobian matrix for each element in the
        mesh"""
        self.get_gauss_points()
        size_conn = self.conn.shape[1]
        vct_tmp = []
        for p in range(size_conn):
            shape_func = []
            for gp in self.gauss_points.gauss_coord:
                el = self.conn[:, p]
                pa, pb, pc = [self.plist[p] for p in el]
                J = np.zeros((2, 2), dtype=np.float64)  # init jacob matrix
                J[0, 0] = -pa[1] + pc[1]
                J[0, 1] = pa[1] - pb[1]
                J[1, 0] = pa[0] - pc[0]
                J[1, 1] = -pa[0] + pb[0]
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                Nav = J.dot(np.array([-1, -1])) / detJ  # Nav = (d(Na)/dx, d(Na)/dy)
                Nbv = J.dot(np.array([1, 0])) / detJ
                Ncv = J.dot(np.array([0, 1])) / detJ
                shape_func.append((Nav, Nbv, Ncv, detJ))
            vct_tmp.append(shape_func)
        self.shape_grad = vct_tmp
