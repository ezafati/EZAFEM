from typing import Tuple, Dict, Type

import numpy as np


def hooke_plane_strain(youn: float, poi: float):
    fact = youn / ((1 - 2 * poi) * (1 + poi))
    D = np.zeros((3, 3), dtype=np.float64)
    D[0, 0] = (1 - poi) * fact
    D[0, 1] = poi * fact
    D[1, 1] = D[0, 0]
    D[1, 0] = D[0, 1]
    D[2, 2] = (1 - 2 * poi) / 2 * fact
    return D


def hooke_plane_stress(youn: float, poi: float):
    fact = youn / (1 - poi ** 2)
    D = np.zeros((3, 3), dtype=np.float64)
    D[0, 0] = 1 * fact
    D[0, 1] = poi * fact
    D[1, 1] = D[0, 0]
    D[1, 0] = D[0, 1]
    D[2, 2] = (1 - poi) / 2 * fact
    return D


def elem_stiff_matrix_tri3(p: int, part: 'Part'):
    probtype = part.probtype
    youn = part.mate.cstprop['Young']
    poi = part.mate.cstprop['Poisson']
    if probtype == 'STRESS':
        D = hooke_plane_stress(youn, poi)
    else:
        D = hooke_plane_strain(youn, poi)
    Ke = np.zeros((part.dim * part.nbvertx, part.dim * part.nbvertx), dtype=np.float64)
    matshpae = np.zeros((3, part.dim * part.nbvertx), dtype=np.float64)  # init shape matrix
    for ind in range(len(part.gauss_points)):
        w = part.gauss_points.weights[ind]
        Nav, Nbv, Ncv, detJ = part.shape_grad[p][ind]
        matshpae[0, 0:6:2] = [Nav[0], Nbv[0], Ncv[0]]
        matshpae[1, 1:6:2] = [Nav[1], Nbv[1], Ncv[1]]
        matshpae[2, 0:6] = [Nav[1], Nav[0], Nbv[1], Nbv[0], Ncv[1], Ncv[0]]
        Ke = Ke + 1 / 2 * w * matshpae.T.dot(D).dot(matshpae) * abs(detJ)
    return Ke


def elem_forc_vect_tri3(p: int, part: 'Part'):
    Nav, Nbv, Ncv, detJ = part.shape_grad[p]
    matshpae = np.zeros((3, 6), dtype=np.float64)  # init shape matrix
    matshpae[0, 0:6:2] = [Nav[0], Nbv[0], Ncv[0]]
    matshpae[1, 1:6:2] = [Nav[1], Nbv[1], Ncv[1]]
    matshpae[2, 0:6] = [Nav[1], Nav[0], Nbv[1], Nbv[0], Ncv[1], Ncv[0]]
