from GlobalObjects.MathUtils import GaussPoints
import os.path

path1, *_ = os.path.split(os.path.abspath('./'))
dbpath = os.path.join(path1, 'src/matdb/gauss.json')

d1 = [(1/3,1/3)]
d2 =(1,)
gp = GaussPoints(eltype='tri3')
gp.get_gauss_points(dbpath)
print(gp.gauss_coord, gp.weights)