import os.path
from GlobalObjects.MatrixObjects import Material

path1, *_ = os.path.split(os.path.abspath('./'))
dbpath = os.path.join(path1, 'src/matdb/matdb.json')


'''add linear elsatic material'''
mate = Material('Linear_Elastic', {'Young': 1E9, 'Poisson': 0.25})
mate.add_material(dbpath)

'''add perfect elatsoplastic material'''

mate = Material('PerfectElastPlastic', {'Young': 1E9, 'Poisson': 0.25}, {'siglimit': 10e6})
mate.add_material(dbpath)

mate = Material('Linear_Elastic')
mate.get_material(dbpath)
print(mate)