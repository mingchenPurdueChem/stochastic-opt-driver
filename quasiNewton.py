import sys
import math
import numpy
from numpy import linalg
from numpy import random

import piny
import quaternion
import stogd

noise_force_constant = 0.05
bohr = 0.529177

def update_coord(coord,force,emp_inv_hess,param_set,step_size):
    hess_scale = param_set[0] # epsilon
    lembda = param_set[1] # lembda
    step_c = param_set[2] # c
    num_atom = param_set[3]
    num_freedom = num_atom*3
    # everything is transport
    coord_atom = coord.reshape((num_atom,3))

    #emp_inv_hess = numpy.identity(num_freedom)
    pt = numpy.dot(force,emp_inv_hess)
    #sys.exit(0)
    st = step_size/step_c*pt
    coord_new = coord+st
    coord_atom_new = coord_new.reshape((num_atom,3))
    coord_atom_new = stogd.match_coord(coord_atom,coord_atom_new)
    coord_new = coord_atom_new.reshape((1,num_freedom))[0]
    #print coord_new
    delt_abs = numpy.fabs(coord_new-coord)
    max_shift = max(delt_abs)*bohr
    mean_shift = numpy.mean(delt_abs)*bohr
    return [coord_new,st,max_shift,mean_shift]

def update_coord_normalize(coord,force,emp_inv_hess,param_set,step_size):
    hess_scale = param_set[0] # epsilon
    lembda = param_set[1] # lembda
    step_c = param_set[2] # c
    num_atom = param_set[3]
    num_freedom = num_atom*3
    # everything is transport
    coord_atom = coord.reshape((num_atom,3))

    #emp_inv_hess = numpy.identity(num_freedom)
    pt = numpy.dot(force,emp_inv_hess)
    #sys.exit(0)
    st = step_size/step_c*pt/linalg.norm(pt)
    coord_new = coord+st
    coord_atom_new = coord_new.reshape((num_atom,3))
    coord_atom_new = stogd.match_coord(coord_atom,coord_atom_new)
    coord_new = coord_atom_new.reshape((1,num_freedom))[0]
    #print coord_new
    delt_abs = numpy.fabs(coord_new-coord)
    max_shift = max(delt_abs)*bohr
    mean_shift = numpy.mean(delt_abs)*bohr
    return [coord_new,st,max_shift,mean_shift]


def diff_force(force_new,force_old,coord_new,coord_old,num_atom):
    # Add an artificial noise to deltaf
    deltaf = force_new-force_old
    deltax = coord_new-coord_old
    num_freedom = num_atom*3
    
    ran_list = random.normal(0.0,noise_force_constant,num_freedom*(num_freedom+1)/2)
    index = 0
    noise_mat = numpy.zeros((num_freedom,num_freedom))
    for i in range(num_freedom):
	for j in range(i,num_freedom):
	    noise_mat[i][j] = ran_list[index]
	    index += 1
	    noise_mat[j][i] = noise_mat[i][j]
    deltaf += numpy.dot(noise_mat,deltax)
    return deltaf

def update_hessian(deltaf,emp_inv_hess,st,param_set,step):
    hess_scale = param_set[0] # epsilon
    lembda = param_set[1] # lembda
    step_c = param_set[2] # c
    num_atom = param_set[3]
    num_freedom = num_atom*3

    yt = -deltaf+lembda*st
    rho_t = 1.0/numpy.dot(st,yt)
    if step==1:
	emp_inv_hess_new = numpy.dot(st,yt)/numpy.dot(yt,yt)*numpy.identity(num_freedom)
    else:
	factor1 = numpy.identity(num_freedom)-rho_t*numpy.outer(st,yt)
	factor2 = numpy.identity(num_freedom)-rho_t*numpy.outer(yt,st)
	emp_inv_hess_new = numpy.dot(numpy.dot(factor1,emp_inv_hess),factor2)+step_c*rho_t*numpy.outer(st,st)
    return emp_inv_hess_new


def sto_bfgs(opt_param_pack):
    coord_old = opt_param_pack[0]
    force_test = opt_param_pack[1]
    force_pkg = opt_param_pack[2]
    step_size = opt_param_pack[3]
    num_atom = opt_param_pack[4]
    file_force_traj = opt_param_pack[5]
    file_coord_traj = opt_param_pack[6]
    force_calc_util = opt_param_pack[7]
    max_shift_conv = opt_param_pack[8]
    mean_shift_conv = opt_param_pack[9]
    max_step = opt_param_pack[10]
    coord_atom_bm = opt_param_pack[11]
    force_old = opt_param_pack[12]
 
    step = 1
    stop_flag = 0 # This one controls whether to stop the optimization
    
    # Get local parameters
    hess_scale = 1.0
    lembda = 0.2
    step_c = 1.0
    param_set = [hess_scale,lembda,step_c,num_atom]
    emp_inv_hess = hess_scale*numpy.identity(num_atom*3)
    while stop_flag==0:
        # Update coordinates
	scale = 1.0
        step_size_now = step_size*scale
        #direct = force_test/linalg.norm(force_test)
	[coord_new,st,maxshift,meanshift] = update_coord(coord_old,force_test,emp_inv_hess,param_set,step_size_now)
        rmsd_now = stogd.test_struct_rmsd(coord_new,coord_atom_bm,num_atom)
        # Update force
        if force_pkg=="piny":
            force_new = piny.lunch(coord_new,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = stogd.add_noise(force_new,num_atom)
	deltaf = diff_force(force_new,force_old,coord_new,coord_old,num_atom)
	emp_inv_hess = update_hessian(deltaf,emp_inv_hess,st,param_set,step)
	#print emp_inv_hess
        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord_new[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force_new))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force_new))
        print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
	#finally, replace coord_old and force_old
	coord_old = coord_new
	force_old = force_new

        #if meanforce_test<0.05:
        if step>100:
            stop_flag = 1
            step_sto_start = step
        # Calculate the convergence flag
        #if (maxshift<max_shift_conv and meanshift<mean_shift_conv) or step>max_step:
        #    stop_flag = 1
        # Update the number of steps
        step += 1
    stop_flag = 0
    while stop_flag==0:
        # Update coordinates
        scale = (step-step_sto_start+10.0)/10.0
        step_size_now = step_size/scale
	if step_size_now<0.01:
	    step_size_now = 0.01
        #direct = force_test/linalg.norm(force_test)
        [coord_new,st,maxshift,meanshift] = update_coord_normalize(coord_old,force_test,emp_inv_hess,param_set,step_size_now)
        rmsd_now = stogd.test_struct_rmsd(coord_new,coord_atom_bm,num_atom)
        # Update force
        if force_pkg=="piny":
            force_new = piny.lunch(coord_new,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = stogd.add_noise(force_new,num_atom)
        deltaf = diff_force(force_new,force_old,coord_new,coord_old,num_atom)
        emp_inv_hess = update_hessian(deltaf,emp_inv_hess,st,param_set,step)
        #print emp_inv_hess
        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord_new[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force_new))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force_new))
        print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
        #finally, replace coord_old and force_old
        coord_old = coord_new
        force_old = force_new

        #if meanforce_test<0.05:
        if step>max_step:
            stop_flag = 1
            step_sto_start = step
        # Calculate the convergence flag
        #if (maxshift<max_shift_conv and meanshift<mean_shift_conv) or step>max_step:
        #    stop_flag = 1
        # Update the number of steps
        step += 1
    







