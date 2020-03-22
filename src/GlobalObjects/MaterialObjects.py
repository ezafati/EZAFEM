import json
import sys

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

    def get_material(self, db, nel, ngp):
        with open(db, 'r') as fdb:
            mates = json.load(fdb)[self.name]
            try:
                if self.cstprop is None:
                    cstprop = mates['cst_properties']
                    ncst = len(cstprop.keys())
                    self.cstprop = np.ndarray((ncst, nel), dtype=np.float32)
                    for p in range(nel):
                        try:
                            self.cstprop[0, p] = cstprop['Young']
                            self.cstprop[1, p] = cstprop['Poisson']
                        except KeyError:
                            sys.exit('Young modulus and the Poisson ratio should be provided ')
                        self.cstprop[2:ncst, p] = [cstprop[key] for key in cstprop.keys() if key != 'Young' and key !=
                                                   'Poisson']
                if self.varprop is None:
                    varprop = mates['var_properties']
                    if varprop is not None:
                        ncst = len(varprop.keys())
                        if ncst:
                            self.varprop = np.ndarray((ngp, ncst, nel), dtype=np.float32)
                            for p in range(nel):
                                for ind in range(ngp):
                                    self.varprop[ind, :, p] = [varprop[key] for key in varprop.keys()]
            except KeyError:
                raise KeyError(f'The provided material name {self.name} not found')

    def affect_properties(self, **kwargs):
        pass
