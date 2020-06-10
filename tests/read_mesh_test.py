from GlobalObjects import *
from GlobalObjects.StiffnessMatrixUtils import compute_def_tensor_tri3, elem_forc_vect_tri3
import os

root_dir, *_ = os.path.split(os.getcwd())
MAIN_YAML_PATH = os.path.join(root_dir, 'src/main.yml')

mymesh = MeshObj(file='mesh.txt')

#mymesh.read_ezamesh()
parts = mymesh.parts
parts[0].initiate()
# print(parts[0].shape_grad)
# print(parts[0])
# grad = part.grad_shape_array()
parts[0].mat_assembly(mtype='stiff')
elem_forc_vect_tri3(10, parts[0])
mymesh.add_boundary(MAIN_YAML_PATH)
