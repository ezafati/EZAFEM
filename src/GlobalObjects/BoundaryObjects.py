from typing import Tuple, List


class Boundary:
    """class describing the boundaries"""

    def __init__(self):
        self.bound_data = {}
        self.point_data = {}


class PerfectInterface:
    def __init__(self, itype: str = None, list_int: List[Tuple['Part', str]] = None):
        self.type = itype
        self.list_int = list_int
        self.link_mats = []

        self.make_link_matrices()

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

