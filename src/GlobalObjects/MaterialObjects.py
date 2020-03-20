import json


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
                if self.cstprop is None:
                    self.cstprop = mates['cst_properties']
                if self.varprop is None:
                    self.varprop = mates['var_properties']
            except KeyError:
                raise KeyError(f'The provided material name {self.name} not found')

    def affect_properties(self, **kwargs):
        pass
