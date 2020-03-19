from typing import Tuple

import numpy as np


def elem_mass_matrix_tri3(el: Tuple[int], part, dens: float, khi: float):
    pt_gauss = (1 / 3, 1 / 3)

    def shape_matrix(xi, nu):
        shmat = np.empty((2, 6))
        shmat[0, 0:6:2] = [1 - xi - nu, xi, nu]
        shmat[1, 1:6:2] = shmat[0, 0:6:2]
        return shmat

    pa, pb, pc = [part.plist[p] for p in el]
    dJ = np.empty((2, 2))
    dJ[0, 0] = -pa[0] + pb[0]
    dJ[0, 1] = -pa[0] + pc[0]
    dJ[1, 0] = -pa[0] + pb[0]
    dJ[1, 1] = -pa[0] + pc[0]

    mat_prod = shape_matrix(*pt_gauss).T.dot(shape_matrix(*pt_gauss))
    return dens * mat_prod * np.linalg.det(dJ)
