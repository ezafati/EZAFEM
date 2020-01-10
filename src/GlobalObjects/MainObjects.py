""""Main GlobalObjects """
import importlib
import json
from typing import Tuple

from scipy.sparse import lil_matrix, coo_matrix


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
        _module = importlib.import_module('GlobalObjects.MatricesUtils')
        assembly_method = _module.__dict__[f'_assembly_{self.mtype}_{self.eltype}']
        assembly_method(self.mat, **kwargs)
        pass
