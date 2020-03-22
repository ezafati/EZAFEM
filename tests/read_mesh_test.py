from GlobalObjects import MeshObj
from GlobalObjects.StiffnessMatrixUtils import compute_def_tensor_tri3, elem_forc_vect_tri3

mymesh = MeshObj()

mymesh.read_ezamesh('mesh.txt')

part = mymesh.parts[0]

grad = part.grad_shape_array()
part.mat_assembly(mtype='stiff')
elem_forc_vect_tri3(10, part)
print(part.eps_array[0,:,10])
#compute_def_tensor_tri3(10, part, 0)
