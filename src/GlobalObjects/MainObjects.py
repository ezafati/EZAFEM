""""Main GlobalObjects """
import json
from typing import Tuple, List

from scipy.sparse import lil_matrix, coo_matrix
from collections import namedtuple

import globalvars

Point2D = namedtuple('Point2D', 'x y')


class Material:
    def __init__(self, name: 'str' = None, cst: 'dict' = None, var: 'dict' = None):
        self.name = name
        self.cstprop = cst
        self.varprop = var

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, cst={self.cstprop}, var={self.varprop})'

    def add_material(self, db: 'json file'):
        with open(db, 'r') as fdb:
            mates = json.load(fdb)
            mates[self.name] = dict()
            mates[self.name]['cst_properties'] = self.cstprop
            mates[self.name]['var_properties'] = self.varprop
        with open(db, 'w') as fdb:
            json.dump(mates, fdb, indent=2)

    def get_material(self, db):
        with open(db, 'r') as fdb:
            mates = json.load(fdb)[self.name]
            try:
                self.cstprop = mates['cst_properties']
                self.varprop = mates['var_properties']
            except KeyError:
                raise KeyError(f'The provided material name {self.name} not found')

    def modify_properties(self, **kwargs):
        pass


class MatrixObj:
    def __init__(self, size: Tuple[int], mtype: str, dim: int, eltype: str):
        self.mat = coo_matrix(size)
        self.mtype = mtype
        self.dim = dim
        self.eltype = eltype

    def assembly(self, **kwargs):
        return eval(f'{self}.assembly_{self.dim}(**{kwargs})')

    def assembly_2d(self, **kwargs):
        nel = globalvars.mesh.conn.size[1]
        nvert = globalvars.mesh.conn.size[0]
        for p in range(nel):
            connel = globalvars.mesh.conn[:, p]
            Kel = eval(f'elem_{self.mtype}_matrix_{self.eltype}({connel}, **{kwargs})')
            for i in range(nvert):
                for j in range(nvert):
                    self.mat[2 * connel[i]:2 * connel[i] + 2, 2 * connel[j]:2 * connel[j] + 2] = self.mat[
                                                                                            2 * connel[i]:2 * connel[
                                                                                                i] + 2,
                                                                                            2 * connel[j]:2 * connel[
                                                                                                j] + 2] + Kel[
                                                                                                          2 * i:2 * i + 2,
                                                                                                          2 * j:2 * j + 2]


