import sys
import math
import numpy
from numpy import linalg
from numpy import random
import quaternion
import piny

bohr = 0.529177

def match_coord(coord_atom_old,coord_atom_new):
    coord_atom_new -= numpy.mean(coord_atom_new,0)
    [eigvalmax,quat] = quaternion.calc_quat(coord_atom_new,coord_atom_old)
    # Rotation matrix from new to old
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    rot_mat = numpy.zeros((3,3))
    # Well, this format is used for atom coord as column, we need a transpose
    rot_mat[0][0] = q0**2+q1**2-q2**2-q3**2
    rot_mat[0][1] = 2.0*(q1*q2-q0*q3)
    rot_mat[0][2] = 2.0*(q1*q3+q0*q2)
    rot_mat[1][0] = 2.0*(q1*q2+q0*q3)
    rot_mat[1][1] = q0**2-q1**2+q2**2-q3**2
    rot_mat[1][2] = 2.0*(q2*q3-q0*q1)
    rot_mat[2][0] = 2.0*(q1*q3-q0*q2)
    rot_mat[2][1] = 2.0*(q2*q3+q0*q1)
    rot_mat[2][2] = q0**2-q1**2-q2**2+q3**2
    rot_mat = rot_mat.transpose()
    
    coord_atom_new = numpy.dot(coord_atom_new,rot_mat)
    return coord_atom_new

def test_struct_rmsd(coord,coord_atom_bm,num_atom):
    coord_atom = coord.reshape((num_atom,3))
    [eigvalmax,quat] = quaternion.calc_quat(coord_atom,coord_atom_bm)
    coord_bm = coord_atom_bm.reshape((1,3*num_atom))[0]
    norm_sq_1 = numpy.dot(coord,coord)
    norm_sq_2 = numpy.dot(coord_bm,coord_bm)
    if norm_sq_1+norm_sq_2<2.0*eigvalmax:
	quaternion.debug(coord_atom,coord_atom_bm)
    rmsd = math.sqrt((norm_sq_1+norm_sq_2-2.0*eigvalmax)/num_atom)*bohr
    return rmsd
 

def update_coord(coord,force,step_size,num_atom):
    num_freedom = num_atom*3
    #step_size_now = step_size
    # Reshape the vectors for convenient
    force_atom = force.reshape((num_atom,3))
    coord_atom = coord.reshape((num_atom,3))

    force_norm = linalg.norm(force)
    #if force_norm>1.0:
    #	force /= force_norm
    #direct = force/force_norm
    direct = force
    coord_new = coord+step_size*direct
    coord_atom_new = coord_new.reshape((num_atom,3))
    coord_atom_new = match_coord(coord_atom,coord_atom_new)
    coord_new = coord_atom_new.reshape((1,num_freedom))[0]
    #print coord_new
    delt_abs = numpy.fabs(coord_new-coord)
    max_shift = max(delt_abs)*bohr
    mean_shift = numpy.mean(delt_abs)*bohr
    return [coord_new,max_shift,mean_shift]

def update_coord_direct(coord,direct,step_size,num_atom):
    num_freedom = num_atom*3
    #step_size_now = step_size
    # Reshape the vectors for convenient
    coord_atom = coord.reshape((num_atom,3))

    coord_new = coord+step_size*direct
    coord_atom_new = coord_new.reshape((num_atom,3))
    coord_atom_new = match_coord(coord_atom,coord_atom_new)
    coord_new = coord_atom_new.reshape((1,num_freedom))[0]
    #print coord_new
    delt_abs = numpy.fabs(coord_new-coord)
    max_shift = max(delt_abs)*bohr
    mean_shift = numpy.mean(delt_abs)*bohr
    return [coord_new,max_shift,mean_shift]

    
def add_noise(force,num_atom):
    # This function is for testing only
    num_freedom = num_atom*3
    # We use different noise level for C and H in testing
    atom_type = [1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0] 

    force_temp = numpy.zeros(num_freedom)
    for i in range(num_freedom):
	force_temp[i] = force[i]
    std = []
    for i in range(num_atom):
	if atom_type==1:
	    for j in range(3):
		std.append(0.05)
	else:
	    for j in range(3):
		std.append(0.02)

    noise_normal = random.normal(0.0,1.0,num_freedom)
    for i in range(num_freedom):
	noise_normal[i] *= std[i]
    #print noise_normal[0]
    force_temp += noise_normal
    
    return force_temp	    
   
def sto_gd(opt_param_pack):
    coord = opt_param_pack[0]
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
    #unzip opt_param_pkg
    step = 1
    stop_flag = 0 # This one controls whether to stop the optimization
    while stop_flag==0:
        # Update coordinates
        step_size_now = step_size
	#direct = force_test/linalg.norm(force_test)
	direct = force_test
        [coord,maxshift,meanshift] = update_coord_direct(coord,direct,step_size_now,num_atom)
        rmsd_now = test_struct_rmsd(coord,coord_atom_bm,num_atom)
        # Update force
        if force_pkg=="piny":
            force = piny.lunch(coord,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = add_noise(force,num_atom)
        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force))
        print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
        #if meanforce_test<0.05:
        if step>200:
            stop_flag = 1
            step_sto_start = step
        # Calculate the convergence flag
        #if (maxshift<max_shift_conv and meanshift<mean_shift_conv) or step>max_step:
        #    stop_flag = 1
        # Update the number of steps
        step += 1
    #step_sto_start = 0
    stop_flag = 0
    step_size *= 1.0
    while stop_flag==0:
        # Update coordinates
        deltan = step-step_sto_start
        #scale = deltan**0.5*math.log(deltan+1.0)**0.6
        #scale = deltan+10.0
        scale = deltan**0.5*math.log(deltan+1)**0.51
        #scale = 1.0
        step_size_now = step_size/scale
	direct = force_test/linalg.norm(force_test)
	#direct = force_test
	[coord,maxshift,meanshift] = update_coord_direct(coord,direct,step_size_now,num_atom)
	rmsd_now = test_struct_rmsd(coord,coord_atom_bm,num_atom)
        # Update force
        if force_pkg=="piny":
            force = piny.lunch(coord,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = add_noise(force,num_atom)
        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force))
        print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
        # Calculate the convergence flag
        if (maxshift<max_shift_conv and meanshift<mean_shift_conv) or step>max_step:
            stop_flag = 1
        # Update the number of steps
        step += 1

def force_correct(force,force_store,force_square_store,num_force_store):
    # update the search direction
    for i in range(1,num_force_store):
	min_force_index = force_square_store.index(min(force_square_store))-1
    force_good = force_store[min_force_index]
    force_square_good = force_square_store[min_force_index]
    force_corr = force+numpy.dot(force,force_good)/force_square_good*force_good
    direct = force_corr/linalg.norm(force_corr)
    return direct


def sto_gd_direct(opt_param_pack,num_force_store):
    #unzip opt_param_pkg
    coord = opt_param_pack[0]
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
    # Initialize force history
    force_store = []
    for i in range(num_force_store):
	force_store.append(numpy.zeros(num_atom*3))
    force_square_store = []
    for i in range(num_force_store):
	force_square_store.append(100000.0)
    force_store[-1] = force_test
    force_square_store[-1] = numpy.dot(force_test,force_test)
    direct = force_test/math.sqrt(force_square_store[0])    

    step = 1
    stop_flag = 0 # This one controls whether to stop the optimization
    while stop_flag==0:
        # Update coordinates
        step_size_now = step_size
        [coord,maxshift,meanshift] = update_coord_direct(coord,direct,step_size_now,num_atom)
        # Update force
        if force_pkg=="piny":
            force = piny.lunch(coord,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = add_noise(force,num_atom)
        # shift the stores
	force_square_store.append(numpy.dot(force_test,force_test))
	force_store.append(force_test)
	force_store.remove(force_store[0])
	force_square_store.remove(force_square_store[0])
        # update the search direction
	direct = force_correct(force_test,force_store,force_square_store,num_force_store)
	
        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force))
        print step_size_now,maxshift,meanshift,maxforce,meanforce
        #if meanforce_test<0.05:
        if step>500:
            stop_flag = 1
            step_sto_start = step
        # Calculate the convergence flag
        #if (maxshift<max_shift_conv and meanshift<mean_shift_conv) or step>max_step:
        #    stop_flag = 1
        # Update the number of steps
        step += 1
    stop_flag = 0
    step_size *= 10.0
    while stop_flag==0:
        # Update coordinates
        deltan = step-step_sto_start
        #scale = deltan**0.5*math.log(deltan+1.0)**0.6
        scale = deltan+10.0
        #scale = deltan
        #scale = 1.0
        step_size_now = step_size/scale
        [coord,maxshift,meanshift] = update_coord(coord,force_test,step_size_now,num_atom)
        # Update force
        if force_pkg=="piny":
            force = piny.lunch(coord,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = add_noise(force,num_atom)
        # shift the stores
        force_square_store.append(numpy.dot(force_test,force_test))
        force_store.append(force_test)
        force_store.remove(force_store[0])
        force_square_store.remove(force_square_store[0])
        # update the search direction
        direct = force_correct(force_test,force_store,force_square_store,num_force_store)

        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force))
        print step_size_now,maxshift,meanshift,maxforce,meanforce
        # Calculate the convergence flag
        if (maxshift<max_shift_conv and meanshift<mean_shift_conv) or step>max_step:
            stop_flag = 1
        # Update the number of steps
        step += 1

def force_mean(force_sum,force,num_force,weight):
    force_sum = force_sum*0.8+force
    weight = weight*0.8+1.0
    force_ave = force_sum/weight
    return [weight,force_ave]

def sto_gd_mean(opt_param_pack,num_force_store):
    #unzip opt_param_pkg
    coord = opt_param_pack[0]
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
    # Initialize force history

    step = 1
    stop_flag = 0 # This one controls whether to stop the optimization
    direct = force_test/linalg.norm(force_test)
    while stop_flag==0:
        # Update coordinates
	scale = 1.0
        step_size_now = step_size/scale
        [coord,maxshift,meanshift] = update_coord_direct(coord,direct,step_size_now,num_atom)
	rmsd_now = test_struct_rmsd(coord,coord_atom_bm,num_atom)
        # Update force
        if force_pkg=="piny":
            force = piny.lunch(coord,force_calc_util)
        # For testing, we add a random noise on deterministic force, we shall comment this after real application
        force_test = add_noise(force,num_atom)
	#print force_sum[0],force_ave[0]
        # update the search direction
	direct = force_test/linalg.norm(force_test)
	
        file_force_traj.write("Step "+str(step)+"\n")
        file_coord_traj.write("Step "+str(step)+"\n")
        # Output this step information
        for i in range(num_atom):
            w_str_coord = ""
            w_str_force = ""
            for j in range(3):
                index = i*3+j
                w_str_coord += str(coord[index]*bohr)+' '
                w_str_force += str(force_test[index])+' '
            w_str_coord += '\n'
            w_str_force += '\n'
            file_coord_traj.write(w_str_coord)
            file_force_traj.write(w_str_force)
        # Report the max coord shift and max force(for test only)
        maxforce = max(numpy.fabs(force))
        meanforce_test = numpy.mean(numpy.fabs(force_test))
        meanforce = numpy.mean(numpy.fabs(force))
        print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
        #if meanforce_test<0.05:
        if step>150:
            stop_flag = 1
        step += 1
    '''
    #direct = force_test
    direct = force_test/linalg.norm(force_test)
    #step_new = 0
    step_size *= 0.1
    for istep in range(max_step-step+1):
        #scale = istep+1
	scale = 1.0
	#scale = (istep+10.0)
	step_size_now = step_size/scale
	coord_old = coord
	[coord,maxshift,meanshift] = update_coord_direct(coord,direct,step_size_now,num_atom)
	rmsd_now = test_struct_rmsd(coord,coord_atom_bm,num_atom)
	# Update force
	if force_pkg=="piny":
	    force = piny.lunch(coord,force_calc_util)
	# For testing, we add a random noise on deterministic force, we shall comment this after real application
	#force_test = add_noise(force,num_atom)+0.95/step_size_now*(coord-coord_old)
	force_test = add_noise(force,num_atom)
	# update the search direction
	direct = force_test/linalg.norm(force_test)
	#direct = force_test
	file_force_traj.write("Step "+str(step)+"\n")
	file_coord_traj.write("Step "+str(step)+"\n")
	# Output this step information
	for i in range(num_atom):
	    w_str_coord = ""
	    w_str_force = ""
	    for j in range(3):
		index = i*3+j
		w_str_coord += str(coord[index]*bohr)+' '
		w_str_force += str(force_test[index])+' '
	    w_str_coord += '\n'
	    w_str_force += '\n'
	    file_coord_traj.write(w_str_coord)
	    file_force_traj.write(w_str_force)
	# Report the max coord shift and max force(for test only)
	maxforce = max(numpy.fabs(force))
	meanforce_test = numpy.mean(numpy.fabs(force_test))
	meanforce = numpy.mean(numpy.fabs(force))
	print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
    
    '''
    nit = 100
    nsample = 10
    direct = force_test/linalg.norm(force_test)
    for it in range(nit):
	if it>0:
	    coord = coord_sum/nsample
	scale = 1.0/(it+1.0)
	step_size_now = step_size*scale
	coord_sum = numpy.zeros(num_atom*3)
	for istep in range(nsample):
	    [coord,maxshift,meanshift] = update_coord_direct(coord,direct,step_size_now,num_atom)
	    coord_sum += coord
	    rmsd_now = test_struct_rmsd(coord,coord_atom_bm,num_atom)
	    # Update force
	    if force_pkg=="piny":
		force = piny.lunch(coord,force_calc_util)
	    # For testing, we add a random noise on deterministic force, we shall comment this after real application
	    force_test = add_noise(force,num_atom)
	    #print force_sum[0],force_ave[0]
	    # update the search direction
	    direct = force_test/linalg.norm(force_test)

	    file_force_traj.write("Step "+str(step)+"\n")
	    file_coord_traj.write("Step "+str(step)+"\n")
	    # Output this step information
	    for i in range(num_atom):
		w_str_coord = ""
		w_str_force = ""
		for j in range(3):
		    index = i*3+j
		    w_str_coord += str(coord[index]*bohr)+' '
		    w_str_force += str(force_test[index])+' '
		w_str_coord += '\n'
		w_str_force += '\n'
		file_coord_traj.write(w_str_coord)
		file_force_traj.write(w_str_force)
	    # Report the max coord shift and max force(for test only)
	    maxforce = max(numpy.fabs(force))
	    meanforce_test = numpy.mean(numpy.fabs(force_test))
	    meanforce = numpy.mean(numpy.fabs(force))
	    print step_size_now,maxshift,meanshift,maxforce,meanforce,rmsd_now
    

 
