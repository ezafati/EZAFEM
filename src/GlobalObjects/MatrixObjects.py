""""Main GlobalObjects """
from typing import Tuple, Type
from scipy.sparse import lil_matrix
import numpy as np
from GlobalObjects.StiffnessMatrixUtils import elem_stiff_matrix_tri3


def mat_assembly_2d(part: Type['Part'], mtype: str, **kwargs):  # to change and be independent of dimension
    """assembly 2D matrices"""
    nel = part.conn.shape[1]
    nvert = part.conn.shape[0]  # number of vertexes by element
    mat = eval(f'part.{mtype}mat')
    for p in range(nel):
        connel = part.conn[:, p]
        Kel = eval(f'elem_{mtype}_matrix_{part.eltype.lower()}(p, part, **kwargs)')
        for i in range(nvert):
            for j in range(nvert):
                mat[2 * connel[i]:2 * connel[i] + 2, 2 * connel[j]:2 * connel[j] + 2] += Kel[2 * i:2 * i + 2,
                                                                                         2 * j:2 * j + 2]


def _vct_assembly_2d(part: Type['Part'], vtype: str, **kwargs):
    """Assembly 1D array"""
    nel = part.conn.shape[1]
    nvert = part.conn.shape[0]  # number of vertexes by element
    vct = eval(f'part.{vtype}vct')
    for p in range(nel):
        connel = part.conn[:, p]
        vel = eval(f'elem_{vtype}_vect_{part.eltype}(p, part, **kwargs)')
        for i in range(nvert):
            vct.mat[2 * connel[i]:2 * connel[i] + 2] = vel[2 * i:2 * i + 2]


class MatrixObj:
    def __init__(self, mtype: str):
        self.mtype = mtype
        self.data = {}  # to store the matrix for each part in the mesh

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
            self.data[instance] = np.ndarray((dim * npt,), dtype=np.float64)
        return self.data.get(instance)

    def __set__(self, instance, value):
        self.data[instance] = value


class LinkMatrixObj:
    def __init__(self):
        self.data = {}  # to store the matrix for each part in the mesh

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
