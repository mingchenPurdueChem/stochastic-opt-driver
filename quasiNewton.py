import sys
import math
import numpy
from numpy import linalg
import piny
import quaternion
import stogd


def update_step(coord,force,emp_inv_hess,param_set):
    step_size = param_set[0] # yita
    hess_scale = param_set[1] # epsilon
    lembda = param_set[2] # lembda
    step_c = param_set[3] # c
    # everything is transport
    pt = -numpy.dot(force,emp_inv_hess)
    st = step_size/step_c*pt
    coord_new = coord+st
    coord_atom_new = coord_new.reshape((num_atom,3))
    coord_atom_new = stogd.match_coord(coord_atom,coord_atom_new)
    coord_new = coord_atom_new.reshape((1,num_freedom))[0]
    #print coord_new
    delt_abs = numpy.fabs(coord_new-coord)
    max_shift = max(delt_abs)*bohr
    mean_shift = numpy.mean(delt_abs)*bohr
    return [coord]

    







