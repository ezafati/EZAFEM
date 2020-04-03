from GlobalObjects import MeshObj
from GlobalObjects.StiffnessMatrixUtils import compute_def_tensor_tri3, elem_forc_vect_tri3

mymesh = MeshObj()

mymesh.read_ezamesh('mesh.txt')

parts = mymesh.parts


print(parts[1], parts[1].conn.shape)

#grad = part.grad_shape_array()
#part.mat_assembly(mtype='stiff')
#elem_forc_vect_tri3(10, part)
#compute_def_tensor_tri3(10, part, 0)
