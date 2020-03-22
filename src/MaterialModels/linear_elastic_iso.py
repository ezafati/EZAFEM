from typing import Tuple


def compute_sigma_internal(eps: 'Array', cstprop: 'Array' = None, varprop: 'Array' = None, probtype: 'Array' = None) \
        -> Tuple['Array', None]:
    from GlobalObjects.StiffnessMatrixUtils import hooke_plane_strain, hooke_plane_stress
    if probtype == 'STRESS':
        D = hooke_plane_stress(cstprop=cstprop)
    else:
        D = hooke_plane_strain(cstprop=cstprop)
    sig = D.dot(eps)
    return sig, None


def hooke_matrix_secant(eps, cstprop, varpropo):
    pass


def hooke_matrix_tangent(eps, cstprop, varpropo):
    pass