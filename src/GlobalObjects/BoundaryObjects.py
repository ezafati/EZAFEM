import math
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
                    try:
                        lbd = part.bound.bound_data.get(bound_name)
                    except KeyError as e:
                        raise Exception(f'boundary {bound_name} not found for part {part.label}:'
                                        f'PLease make sure that the appropriate boundaries are well specified')
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
        self.list_int = list_int  # (part, named-boundary)

    def __repr__(self):
        return f'{self.__class__.__name__}(itype={self.type}, list_int={self.list_int})'

    def bound_reorder(self):
        """ re-order the named boundary of one of the two
        sub-domains """
        prt1, prt2 = [p[0] for p in self.list_int]
        lbd1, lbd2 = [p[0].bound.bound_data[p[1]] for p in self.list_int]
        try:
            assert len(lbd1) == len(lbd1)
        except AssertionError:
            raise Exception(f'non matching perfect interface between {prt1.label} and {prt2.label}')

        for npt1 in lbd1:
            npt2, *_ = filter(self.check_point(prt1, prt2, npt1), lbd2)
            lbd2.remove(npt2)
            lbd2.append(npt2)

    @staticmethod
    def check_point(prt1, prt2, npt1):
        epsilon = 0.000001

        def check_equal(npt2):
            dist = math.sqrt(prt1.plist[npt1][0] ** 2 + prt1.plist[npt1][1] ** 2)
            abs_test = abs(prt1.plist[npt1][0] - prt2.plist[npt2][0]) / dist
            ord_test = abs(prt1.plist[npt1][1] - prt2.plist[npt2][1]) / dist
            return (abs_test <= epsilon) and (ord_test <= epsilon)

        return check_equal

    def make_link_matrices(self):
        """C=return tuple contains the link matrices
        corresponding to the interface A-B"""
        self.bound_reorder()
        bol = -1
        mlink1, mlink2 = self.link_mat  # mlink = Tuple ==> (part, link-matrix)
        for inter in self.list_int:
            part = inter[0]
            bound_name = inter[1]
            bound_data = part.bound.bound_data[bound_name]
            if part.__class__.__name__ == 'SolidPart':
                if part is mlink1[0]:
                    mat = mlink1[1]
                    count = 0
                    dim = part.dim
                    for npt in bound_data:
                        mat[count, dim * npt] = bol
                        mat[count + 1, dim * npt + 1] = bol
                        count += 2
                else:
                    mat = mlink2[1]
                    count = 0
                    dim = part.dim
                    for npt in bound_data:
                        mat[count, dim * npt] = bol
                        mat[count + 1, dim * npt + 1] = bol
                        count += 2
            bol *= -1

