from typing import Tuple, Dict

import numpy as np

import globalvars


def hooke_plane_strain(youn: float, poi: float):
    fact = youn / ((1 - 2*poi)*(1+poi))
    D = np.zeros((3, 3), dtype=np.float32)
    D[0, 0] = (1-poi) * fact
    D[0, 1] = poi * fact
    D[1, 1] = D[0, 0]
    D[1, 0] = D[0, 1]
    D[2, 2] = (1 - 2*poi) / 2 * fact
    return D


def hooke_plane_stress(youn: float, poi: float):
    fact = youn / (1 - poi ** 2)
    D = np.zeros((3, 3), dtype=np.float32)
    D[0, 0] = 1 * fact
    D[0, 1] = poi * fact
    D[1, 1] = D[0, 0]
    D[1, 0] = D[0, 1]
    D[2, 2] = (1 - poi) / 2 * fact
    return D


def elem_stiff_matrix_tri3(el: Tuple[int], cstprop: Dict[str, float]):
    probtype = globalvars.mesh.probtype
    youn = cstprop['Young']
    poi = cstprop['Poisson']
    if probtype == 'STRESS':
        D = hooke_plane_stress(youn, poi)
    else:
        D = hooke_plane_strain(youn, poi)
    pa, pb, pc = [globalvars.mesh.plist[p] for p in el]
    J = np.zeros((2, 2), dtype=np.float32)  # init jacob matrix
    J[0, 0] = -pa[1] + pb[1]
    J[0, 1] = pa[1] - pc[1]
    J[1, 0] = pa[0] - pb[0]
    J[1, 1] = -pa[0] + pc[0]
    detJ = np.linalg.det(J)

    Nav = J * np.array([-1, -1]).T / detJ
    Nbv = J * np.array([1, 0]).T / detJ
    Ncv = J * np.array([0, 1]).T / detJ

    matshpae = np.zeros((3, 6), dtype=np.float32)  # init shape matrix
    matshpae[0, 0:6:2] = [Nav[0], Nbv[0], Ncv[0]]
    matshpae[1, 1:6:2] = [Nav[1], Nbv[1], Ncv[1]]
    matshpae[2, 0:6] = [Nav[0], Nav[1], Nbv[0], Nbv[1], Ncv[0], Ncv[1]]

    Ke = matshpae.T * D * matshpae * abs(detJ)
    return Ke
