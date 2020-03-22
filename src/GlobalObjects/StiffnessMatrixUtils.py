from typing import Tuple, Dict, Type

import numpy as np
from itertools import chain
import importlib


def hooke_plane_strain(cstprop: 'Array', **kwargs):
    youn = cstprop[0]
    poi = cstprop[1]
    fact = youn / ((1 - 2 * poi) * (1 + poi))
    D = np.zeros((3, 3), dtype=np.float64)
    D[0, 0] = (1 - poi) * fact
    D[0, 1] = poi * fact
    D[1, 1] = D[0, 0]
    D[1, 0] = D[0, 1]
    D[2, 2] = (1 - 2 * poi) / 2 * fact
    return D


def hooke_plane_stress(cstprop: 'Array', **kwargs):
    youn = cstprop[0]
    poi = cstprop[1]
    fact = youn / (1 - poi ** 2)
    D = np.zeros((3, 3), dtype=np.float64)
    D[0, 0] = 1 * fact
    D[0, 1] = poi * fact
    D[1, 1] = D[0, 0]
    D[1, 0] = D[0, 1]
    D[2, 2] = (1 - poi) / 2 * fact
    return D


def elem_stiff_matrix_tri3(p: int, part: 'Part', **kwargs) -> Type[np.ndarray]:
    """"Compute the element stiffness matrix
    part: instance of class Part
    ind: index of the point gauss
    return ndarray
        """
    probtype = part.probtype
    Ke = np.zeros((part.dim * part.nbvertx, part.dim * part.nbvertx), dtype=np.float64)
    matshpae = np.zeros((3, part.dim * part.nbvertx), dtype=np.float64)
    try:
        update_type = kwargs['UPDATE']  # update_type take SECANT or TANGENT
        law_module = importlib.import_module(f'MaterialModels.{part.mate.name.lower()}')
        hooke_func = law_module.__dict__[
            f'hooke_matrix_{update_type.lower()}']  # update type will be handled at this level
        for ind in range(len(part.gauss_points)):
            eps = part.eps_array[ind, :, p]
            D = hooke_func(eps, part.mate.cstprop[:, p], part.mate.varprop[ind, :, p])
            w = part.gauss_points.weights[ind]
            Nav, Nbv, Ncv, detJ = part.shape_grad[p][ind]
            matshpae[0, 0:6:2] = [Nav[0], Nbv[0], Ncv[0]]
            matshpae[1, 1:6:2] = [Nav[1], Nbv[1], Ncv[1]]
            matshpae[2, 0:6] = [Nav[1], Nav[0], Nbv[1], Nbv[0], Ncv[1], Ncv[0]]
            Ke = Ke + 1 / 2 * w * matshpae.T.dot(D).dot(matshpae) * abs(detJ)
    except KeyError:
        if probtype == 'STRESS':
            D = hooke_plane_stress(cstprop=part.mate.cstprop[:, p])
        else:
            D = hooke_plane_strain(cstprop=part.mate.cstprop[:, p])
        # init shape matrix
        for ind in range(len(part.gauss_points)):
            w = part.gauss_points.weights[ind]
            Nav, Nbv, Ncv, detJ = part.shape_grad[p][ind]
            matshpae[0, 0:6:2] = [Nav[0], Nbv[0], Ncv[0]]
            matshpae[1, 1:6:2] = [Nav[1], Nbv[1], Ncv[1]]
            matshpae[2, 0:6] = [Nav[1], Nav[0], Nbv[1], Nbv[0], Ncv[1], Ncv[0]]
            Ke = Ke + 1 / 2 * w * matshpae.T.dot(D).dot(matshpae) * abs(detJ)
    return Ke


def elem_forc_vect_tri3(p: int, part: 'Part'):
    """"Compute the element force vector
     p: index of the element in the connectivity matrix
    part: instance of class Part"""
    law_module = importlib.import_module(f'MaterialModels.{part.mate.name.lower()}')
    law_func = law_module.__dict__['compute_sigma_internal']
    for ind in range(len(part.gauss_points)):
        eps = compute_def_tensor_tri3(p, part, ind)
        part.eps_array[ind, :, p] = eps
        print(type(eps), type(part.mate.cstprop[:, p]), type(part.probtype))
        try:
            result = law_func(eps=eps, cstprop=part.mate.cstprop[:, p], varprop=part.mate.varprop[ind, :, p],
                              probtype=part.probtype)
        except TypeError:
            result = law_func(eps=eps, cstprop=part.mate.cstprop[:, p], probtype=part.probtype)
        print(result)


def compute_def_tensor_tri3(p: int, part: 'Part', ind: int) -> Type[np.array]:
    """Compute the deformation vector in the Voigt notation
    p: index of the element in the connectivity matrix
    part: instance of class Part
    ind: index of the point gauss """
    el = part.conn[:, p]
    listindx = map(lambda l: [part.dim * l, part.dim * l + 1], el)
    eldisp = part.dispvct[list(chain(*listindx))]
    matshpae = np.zeros((3, part.dim * part.nbvertx), dtype=np.float64)
    Nav, Nbv, Ncv, detJ = part.shape_grad[p][ind]
    matshpae[0, 0:6:2] = [Nav[0], Nbv[0], Ncv[0]]
    matshpae[1, 1:6:2] = [Nav[1], Nbv[1], Ncv[1]]
    matshpae[2, 0:6] = [Nav[1], Nav[0], Nbv[1], Nbv[0], Ncv[1], Ncv[0]]
    return matshpae.dot(eldisp)
