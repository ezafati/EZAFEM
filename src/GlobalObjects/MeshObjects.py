from typing import List, Tuple, Type
from GlobalObjects.MatrixObjects import MatrixObj, mat_assembly_2d, VectObject
from GlobalObjects.MathUtils import GaussPoints
from GlobalObjects.MaterialObjects import Material
from GlobalObjects.BoundaryObjects import Boundary, PerfectInterface
import numpy as np
import os.path
import yaml

MAIN_YAML_PATH = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'main.yml')
MATERIAL_DB = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'matdb/matdb.json')
GAUSS_POINTS_JSON = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'matdb/gauss.json')


class MeshObj(object):

    def __init__(self, dim: int = 2):
        self.dim = dim
        self.parts = []
        self.boundaries = None

    def get_parts(self):
        with open(MAIN_YAML_PATH, 'r') as f:
            d = yaml.load(f.read(), Loader=yaml.Loader)
        parts = d['parts']
        for part in parts:
            if part['physic'] == 'SOLID':
                spart = SolidPart(label=part['part'], dim=self.dim, probtype=part['probtype'],
                                  eltype=part['mesh']['element_type'])
                spart.probtype = part['probtype']
                self.parts.append(spart)
            elif part['physic'] == 'FLUID':
                return NotImplemented

    def get_part_plist(self, part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1].strip() == part.label:
                            if next(f).split(' ')[1].strip() == part.eltype:
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

    @staticmethod
    def get_part_topology(part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1].strip() == part.label:
                            if next(f).split(' ')[1].strip() == part.eltype:
                                while True:
                                    line = next(f)
                                    if line.strip() == 'TOPOLOGY':
                                        size = int(next(f).split(' ')[1])
                                        part.conn = np.ndarray(shape=(part.nbvertx, size), dtype=np.int)
                                        for nel in range(size):
                                            el = next(f).split(' ')
                                            part.conn[:, nel] = [int(el[p + 1]) for p in range(part.nbvertx)]
                                        break
                                break
                    except IndexError:
                        pass
            except StopIteration:
                pass

    @staticmethod
    def get_part_boundary(part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1].strip() == part.label:
                            if next(f).split(' ')[1].strip() == part.eltype:
                                while True:
                                    line = next(f)
                                    if line.strip() == 'NAMED_BOUNDARIES':
                                        size = int(next(f).split(' ')[1])
                                        if not isinstance(part.bound, Boundary):
                                            part.bound = Boundary()
                                        for _ in range(size):
                                            bd = next(f).strip()
                                            lpt = next(f).split(',')
                                            part.bound.bound_data[bd] = [int(p) for p in lpt]
                                        break
                                break
                    except IndexError:
                        pass
            except StopIteration:
                pass

    @staticmethod
    def get_part_points(part, file):
        with open(file, 'r') as f:
            try:
                while True:
                    line = next(f)
                    try:
                        if line.split(' ')[1].strip() == part.label:
                            if next(f).split(' ')[1].strip() == part.eltype:
                                while True:
                                    line = next(f)
                                    if line.strip() == 'NAMED_POINTS':
                                        size = int(next(f).split(' ')[1])
                                        if not isinstance(part.bound, Boundary):
                                            part.bound = Boundary()
                                        for _ in range(size):
                                            pt, *_, npt = next(f).split(' ')
                                            part.bound.point_data[pt] = int(npt)
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
        self.get_parts()
        for part in self.parts:
            self.get_part_plist(part, file)
            self.get_part_topology(part, file)
            self.get_part_boundary(part, file)
            self.get_part_points(part, file)

    def add_boundary(self, file):
        """"add boundaries specified in
        main.yml"""
        self.boundaries = []
        with open(file, 'r') as f:
            d = yaml.load(f.read(), Loader=yaml.Loader)
        try:
            bounds = d['boundary']
            for bound in bounds:
                bd = eval(f"{bound.get('type')}()")
                parts = bound['parts']
                for part in parts:
                    prt, = filter(lambda p: p.label == part['part'], self.parts)
                    bd_label = part.get('bound')
                    if bd.list_int is None:
                        bd.list_int = []
                    bd.list_int.append((prt, bd_label))
                print(bd)
        except (KeyError, NameError) as e:
            raise Exception('appropriate boundaries should be specified in main.yml file: ', e)


class SolidPart(MeshObj):

    def __new__(cls, label: str = 'PART', plist: List[Tuple[int]] = None, conn: 'Array' = None, gard: 'Array' = None,
                mate: Type[Material] = None, dim: int = 2, eltype: str = None, probtype: str = None,
                nbvertx: int = None, defoarray: 'Array' = None):

        cls.stiffmat = MatrixObj(mtype='stiff')
        cls.massmat = MatrixObj(mtype='mass')
        cls.dispvct = VectObject(vtype='disp')
        cls.forcvct = VectObject(vtype='forc')
        instance = super().__new__(cls)
        return instance

    def __init__(self, label: str = 'PART', plist: List[Tuple[int]] = None, conn: 'Array' = None, gard: 'Array' = None,
                 mate: Type[Material] = None, dim: int = 2, eltype: str = None, probtype: str = None,
                 nbvertx: int = None, defoarray: 'Array' = None):
        super().__init__(dim)
        self.label = label
        self.eltype = eltype
        self.nbvertx = nbvertx
        self.plist = plist
        self.conn = conn
        self.gard = gard
        self.mate = mate
        self.shape_grad = gard
        self.defo_array = defoarray
        self.gauss_points = None
        self.eps_array = None
        self.bound = None

        if self.eltype == 'TRI3':
            self.nbvertx = 3
        elif self.eltype == 'TRI6':
            self.nbvertx = 6

    def __repr__(self):
        return f'{self.__class__.__name__}(label={self.label}, eltype={self.eltype}, mate={self.mate}, ' \
               f'probtype={self.probtype}, dim={self.dim}, bound={self.bound})'

    def initiate(self):
        """initialize the solid part"""
        self.grad_shape_array()
        self.get_part_material()
        self.initiliaze_eps_array()

    def initiliaze_eps_array(self):
        ngp = len(self.gauss_points)
        nel = self.conn.shape[1]
        self.eps_array = np.ndarray((ngp, 4, nel), dtype=np.float64)

    def get_gauss_points(self):
        self.gauss_points = GaussPoints(eltype=self.eltype.lower())
        self.gauss_points.get_gauss_points(GAUSS_POINTS_JSON)

    def get_part_material(self):  # to modify
        with open(MAIN_YAML_PATH, 'r') as f:
            d = yaml.load(f.read(), Loader=yaml.Loader)
        try:
            part_list = d['parts']
            for p in part_list:
                if p['part'] == self.label:
                    self.mate = Material()
                    self.mate.name = p['material']['type']
                    self.mate.get_material(MATERIAL_DB, self.conn.shape[1], len(self.gauss_points))
        except KeyError:
            print("DEFAULT MATERIAL WILL BE ASSIGNED TO THE PARTS")
            self.mate = Material(name='Linear_Elastic')
            self.mate.get_material(MATERIAL_DB, self.conn.shape[1])

    def mat_assembly(self, mtype, **kwargs):
        """assembly matrix according to the
        matrix type"""
        mat = eval(f'self.{mtype}mat')
        eval(f'mat_assembly_{self.dim}d(self, mtype, **kwargs)')

    def grad_shape_array(self):
        """"compute the gradient shape array
        """
        eval(f'self.grad_shape_array_{self.eltype.lower()}()')

    def grad_shape_array_tri3(self):
        """Compute the strain-displacement components and the
        determinant of the Jacobian matrix for each element in the
        mesh"""
        self.get_gauss_points()
        size_conn = self.conn.shape[1]
        vct_tmp = []
        for p in range(size_conn):
            shape_func = []
            for _ in self.gauss_points.gauss_coord:
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
