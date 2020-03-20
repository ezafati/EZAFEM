import json


class GaussPoints:
    def __init__(self, eltype = None, gc=None, weights=None):
        self.eltype = eltype
        self.gauss_coord = gc
        self.weights = weights

    def __repr__(self):
        return f'{self.__class__.__name__}(gauss_set={self.gauss_coord}, weights={self.weights})'

    def __len__(self):
        return len(self.weights)

    def add_gauss_points(self, jlist: 'json file'):
        with open(jlist, 'r') as f:
            mates = json.load(f)
            mates[self.eltype] = dict()
            mates[self.eltype]['gauss_coord'] = self.gauss_coord
            mates[self.eltype]['weights'] = self.weights
        with open(jlist, 'w') as f:
            json.dump(mates, f, indent=2)

    def get_gauss_points(self, jlist: 'json file'):
        with open(jlist, 'r') as f:
            mates = json.load(f)[self.eltype]
            try:
                if self.gauss_coord is None:
                    self.gauss_coord = mates['gauss_coord']
                if self.weights is None:
                    self.weights = mates['weights']
            except KeyError:
                raise KeyError(f'The provided material name {self.eltype} not found')


