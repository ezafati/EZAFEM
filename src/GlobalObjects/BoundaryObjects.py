from typing import Tuple, List
import numpy as np
from scipy.sparse import lil_matrix


class Boundary:
    """class describing the boundaries"""

    def __init__(self):
        self.bound_data = {}
        self.point_data = {}


class LinkMatrix:
    """ descriptor to store the links matrices
    for each interface SOLID-SOLID or SOLID-FLUID
    or FLUID-FLUID"""
    def __init__(self):
        self._data = {}  # to store the matrix for each part in the mesh

    @property
    def data(self):
        return self._data

    def __get__(self, instance, owner):
        if not instance:
            return self
        if instance not in self.data:
            if len(instance.list_int) == 2:
                self.data[instance] = list()
                for inter in instance.list_int:
                    part, bound_name = inter
                    dim = part.dim
                    npt = part.plist.shape[0]
                    lbd = part.bound.bound_data.get(bound_name)
                    if part.__class__.__name__ == 'SolidPart':
                        mlink = lil_matrix((dim * len(lbd), dim * npt), dtype=np.float64)
                    else:
                        mlink = None
                    self.data[instance].append((part, mlink))
        return self.data.get(instance)

    def __set__(self, instance, value):
        self.data[instance] = value


class PerfectInterface:
    link_mat = LinkMatrix()

    def __init__(self, itype: str = None, list_int: List[Tuple['Part', str]] = None):
        self.type = itype
        self.list_int = list_int
        self.link_mats = []
        # self.make_link_matrices()

    def __repr__(self):
        return f'{self.__class__.__name__}(itype={self.type}, list_int={self.list_int})'

    def make_link_matrices(self):
        """C=return tuple contains the link matrices
        corresponding to the interface A-B"""
        for inter in self.list_int:
            part = inter[0]
            bound_name = inter[1]
            if part.__class__.__name__ == 'SolidPart':
                pass
