from typing import Tuple

import numpy as np


import globalvars


def elem_mass_matrix_tri3(el: Tuple[int], dens: float, khi: float):
    pt_gauss = (1 / 3, 1 / 3)

    def shape_matrix(xi, nu):
        shmat = np.empty((2, 6))
        shmat[0, 0:6:2] = [1 - xi - nu, xi, nu]
        shmat[1, 1:6:2] = shmat[0, 0:6:2]
        return shmat

    pa, pb, pc = [globalvars.mesh.plist[p] for p in el]
    dJ = np.empty((2, 2))
    dJ[0, 0] = -pa.x + pb.x
    dJ[0, 1] = -pa.x + pc.x
    dJ[1, 0] = -pa.y + pb.y
    dJ[1, 1] = -pa.y + pc.y

    mat_prod = shape_matrix(*pt_gauss).T.dot(shape_matrix(*pt_gauss))
    return dens * mat_prod * np.linalg.det(dJ)
