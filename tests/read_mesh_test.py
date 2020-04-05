from GlobalObjects import MeshObj
from GlobalObjects.StiffnessMatrixUtils import compute_def_tensor_tri3, elem_forc_vect_tri3

mymesh = MeshObj()

mymesh.read_ezamesh('mesh.txt')

parts = mymesh.parts



parts[0].initiate()
#print(parts[0].shape_grad)
#print(parts[0])
#grad = part.grad_shape_array()
parts[0].mat_assembly(mtype='stiff')
elem_forc_vect_tri3(10, parts[0])

#compute_def_tensor_tri3(10, part, 0)
