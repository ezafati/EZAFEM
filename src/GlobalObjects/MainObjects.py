""""Main GlobalObjects """
import json
from typing import Tuple

from scipy.sparse import coo_matrix

from GlobalObjects import initialize_deco
import numpy as np


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

    def affect_properties(self, **kwargs):
        pass


class Boundary:
    """class describing the boundray conditions"""
    pass


class Interface:
    def __init__(self, itype: str = None, dom1: Tuple[str] = None, dom2: Tuple[str] = None):
        self.type = itype
        self.domA = dom1
        self.domB = dom2

    def make_link_matrices(self):
        """C=return tuple contains the link matrices
        corresponding to the interface A-B"""
        return 'not implemented'


@initialize_deco
class MatrixObj:
    def __init__(self, mtype: str):
        self.mtype = mtype
        self.mat = coo_matrix((self.npts, self.npts), dtype=np.float32)

    def assembly(self, **kwargs):
        return eval(f'{self}.assembly_{self.dim}(**{kwargs})')

    def assembly_2d(self, **kwargs):
        nel = self.nel
        nvert = self.npts
        for p in range(nel):
            connel = self.conn
            Kel = eval(f'elem_{self.mtype}_matrix_{self.eltype}({connel}, **{kwargs})')
            for i in range(nvert):
                for j in range(nvert):
                    self.mat[2 * connel[i]:2 * connel[i] + 2, 2 * connel[j]:2 * connel[j] + 2] = self.mat[
                                                                                                 2 * connel[i]:2 *
                                                                                                               connel[
                                                                                                                   i] + 2,
                                                                                                 2 * connel[j]:2 *
                                                                                                               connel[
                                                                                                                   j] + 2] + Kel[
                                                                                                                             2 * i:2 * i + 2,
                                                                                                                             2 * j:2 * j + 2]


@initialize_deco
class VectObject:
    def __init__(self, vtype: str):
        self.vtype = vtype
        self.mat = coo_matrix((self.npts, 1), dtype=np.float32)

    def assembly(self, **kwargs):
        return eval(f'{self}.assembly_{self.dim}(**{kwargs})')

    def assembly_2d(self, **kwargs):
        nel = self.nel
        nvert = self.npts
