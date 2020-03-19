from GlobalObjects import MeshObj


mymesh = MeshObj()

mymesh.read_ezamesh('mesh.txt')

part = mymesh.parts[0]

grad = part.grad_shape_array()
part.mat_assembly('stiff')
