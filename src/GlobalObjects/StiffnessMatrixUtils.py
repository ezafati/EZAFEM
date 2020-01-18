from typing import Tuple, Dict

import numpy as np

import globalvars


def elem_stiff_matrix_tri3(el: Tuple[int], cstprop: Dict[str, float]):
    probtype = globalvars.mesh.probtype
    youn = cstprop['Young']
    poi = cstprop['Poisson']
    if probtype == 'STRESS':
        fact = youn / (1 - nu ** 2)
        D = np.zeros((3, 3), dtype=np.float32)
        D[0, 0] = 1 * fact
        D[0, 1] = poi * fact
        D[1, 1] = D[0, 0]
        D[1, 0] = D[0, 1]
        D[2, 2] = (1 - poi) / 2 * fact
    J = np.zeros((2, 2), dtype=np.float32)  # init jacob matrix
    matshpae = np.zeros((3, 6), dtype=np.float32)  # init shape matrix
