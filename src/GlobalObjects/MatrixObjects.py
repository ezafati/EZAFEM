""""Main GlobalObjects """
import json
from typing import Tuple, Type
from scipy.sparse import lil_matrix
import numpy as np
from GlobalObjects.StiffnessMatrixUtils import elem_stiff_matrix_tri3


def _mat_assembly_2d(part, mtype, **kwargs):
    nel = part.conn.shape[1]
    nvert = part.conn.shape[0]  # number of vertexes by element
    mat = eval(f'part.{mtype}mat')
    for p in range(nel):
        connel = part.conn[:, p]
        Kel = eval(f'elem_{mtype}_matrix_{part.eltype}(p, part, **kwargs)')
        for i in range(nvert):
            for j in range(nvert):
                mat[2 * connel[i]:2 * connel[i] + 2, 2 * connel[j]:2 * connel[j] + 2] = mat[
                                                                                        2 * connel[i]:2 * connel[i] + 2,
                                                                                        2 * connel[j]:2 * connel[
                                                                                            j] + 2] + Kel[
                                                                                                      2 * i:2 * i + 2,
                                                                                                      2 * j:2 * j + 2]


def _vct_assembly_2d(part, vtype, **kwargs):
    nel = part.conn.shape[1]
    nvert = part.conn.shape[0]  # number of vertexes by element
    vct = eval(f'part.{vtype}vct')
    for p in range(nel):
        connel = part.conn[:, p]
        vel = eval(f'elem_{vtype}_vect_{part.eltype}(p, part, **kwargs)')
        for i in range(nvert):
            vct.mat[2 * connel[i]:2 * connel[i] + 2] = vel[2 * i:2 * i + 2]


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
            npt = instance.plist.shape[0]
            dim = instance.dim
            self.data[instance] = lil_matrix((dim * npt, dim * npt), dtype=np.float64)
        return self.data.get(instance)

    def __set__(self, instance, value):
        self.data[instance] = value


class VectObject:
    def __init__(self, vtype: str):
        self.vtype = vtype  # vector type (force, ...)
        self.data = {}

    def __get__(self, instance, owner):
        if not instance:
            return self
        if instance not in self.data:
            npt = instance.plist.shape[0]
            dim = instance.dim
            self.data[instance] = lil_matrix((dim * npt, 1), dtype=np.float64)
        return self.data.get(instance)

    def __set__(self, instance, value):
        self.data[instance] = value
