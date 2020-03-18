""""Main GlobalObjects """
import json
from typing import Tuple, Type
from scipy.sparse import lil_matrix
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


class MatrixObj:
    def __init__(self, mtype: str):
        self.mtype = mtype
        self.data = {}

    def __get__(self, instance, owner):
        if not instance:
            return self
        if instance not in self.data:
            npt = instance.plist.size[0]
            dim = instance.dim
            self.data[instance] = lil_matrix((dim * npt, dim * npt), dtype=np.float64)
        return self.data.get(instance)

    def __set__(self, instance, value):
        self.data[instance] = value

    def assembly(self, part, **kwargs):
        dim = part.dim
        return eval(f'{self}.assembly_{dim}({part},**{kwargs})')

    def assembly_2d(self, part, **kwargs):
        nel = part.conn.size[1]
        nvert = part.nbvertx  # number of vertexes by element
        for p in range(nel):
            connel = part.conn[:, p]
            Kel = eval(f'elem_{self.mtype}_matrix_{part.eltype}({p}, {part}, **{kwargs})')
            for i in range(nvert):
                for j in range(nvert):
                    self.data[part][2 * connel[i]:2 * connel[i] + 2, 2 * connel[j]:2 * connel[j] + 2] = self.data[part][
                                                                                                        2 * connel[
                                                                                                            i]:2 *
                                                                                                               connel[
                                                                                                                   i] + 2,
                                                                                                        2 * connel[
                                                                                                            j]:2 *
                                                                                                               connel[
                                                                                                                   j] + 2] + Kel[
                                                                                                                             2 * i:2 * i + 2,
                                                                                                                             2 * j:2 * j + 2]


"""class VectObject:
    def __init__(self, vtype: str):
        self.vtype = vtype  # vector type (force, ...)
        self.mat = lil_matrix((self.npts, 1), dtype=np.float32)

    def assembly(self, **kwargs):
        return eval(f'{self}.assembly_{self.dim}(**{kwargs})')

    def assembly_2d(self, **kwargs):
        nel = self.nel
        nvert = self.nbvertx
        for p in range(nel):
            connel = self.conn[:, p]
            vel = eval(f'elem_{self.vtype}_vect_{self.eltype}({p}, {mesh}, **{kwargs})')
            for i in range(nvert):
                self.mat[2 * connel[i]:2 * connel[i] + 2] = vel[2 * i:2 * i + 2]"""
