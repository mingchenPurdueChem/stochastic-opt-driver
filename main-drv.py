import sys
import math
import shutil
import os
import numpy
from numpy import linalg
# This is to use stochastic gradient descent
import stogd
# This is the module for specific package to calculate force
import piny

#########################################################
# This is the main driver to do stochastic optimization #
# working with stochastic DFT or any other stochastic   #
# electronic structure calculations. It require a       #
# configuration file to read in, with all options in it.#
# Besides I/O, all units are atomic units. The unit of  #
# coordinates should be in A for I/O.                   #                             
#########################################################


# First, initialize default options
bohr = 0.529177
max_step = 1000
step_size = 0.1 # A
opt_type = "gradient"
force_pkg = ""
force_traj_name = "force-traj"
coord_traj_name = "coord-traj"
init_coord_name = "init-coord"
working_folder = ""
binary_command = ""
force_file_name = ""
num_atom = 0
# These two are criteria of convergence, I can have more
max_shift_conv = 0.001
mean_shift_conv = 0.0001
# These are special parameters for different algorithm
num_force_store = 10 # for optimal direction stochastic gd

fconfig = open(sys.argv[1],'r') 
for line in fconfig:
    line_elem = line.split()
    if len(line_elem)>0:
	opt_key = line_elem[0]
	# I only put two, but we can have more
	if opt_key=="max_step":
	    max_step = int(line_elem[1])
	if opt_key=="opt_type":
	    opt_type = line_elem[1]
	if opt_key=="init-name":
	    init_coord_name = line_elem[1]
	if opt_key=="force-traj-name":
	    force_traj_name = line_elem[1]	    
	if opt_key=="coord-traj-name":
	    coord_traj_name = line_elem[1]
	if opt_key=="num_atom":
	    num_atom = int(line_elem[1])
	if opt_key=="force_package":
	    force_pkg = line_elem[1]
	if opt_key=="max_shift":
	    max_shift_conv = float(line_elem[1])
	if opt_key=="mean_shift":
	    mean_shift_conv = float(line_elem[1])
	if opt_key=="working_folder":
	    working_folder = line_elem[1]
	if opt_key=="binary_command":
	    for i in range(1,len(line_elem)):
	        binary_command += line_elem[i]+' '
	if opt_key=="force_file_name":
	    force_file_name = line_elem[1]
	if opt_key=="step_size":
	    step_size = float(line_elem[1])
	if opt_key=="num_force_store":
	    num_force_store = int(line_elem[1])

fconfig.close()

# Give a brief check of mistakes
if num_atom<=0:
    print "Please input a valid number of atoms, now you have "+str(num_atom)+" atoms"
    sys.exit(0)
if len(force_pkg)==0:
    print "Please specify a package to evaluate the force!"
    sys.exit(0)
if len(working_folder)==0:
    print "Please specify a working folder!"
    sys.exit(0)
if len(binary_command)==0:
    print "Please specify the binary name and path you want to use to calculate force!"
    sys.exit(0)
if len(force_file_name)==0:
    print "Please specify the force file outputed from the package!"
    sys.exit(0)
step_size /= bohr

force_calc_util = [num_atom,working_folder,binary_command,force_file_name]
# Initialize the IO
file_coord_read = open(init_coord_name,'r')
file_coord_traj = open(coord_traj_name,'w')
file_force_traj = open(force_traj_name,'w')
# Get initial coordinates
coord = []
for line in file_coord_read:
    line_elem = line.split()
    if len(line_elem)==3:
	#coord.append(float(line_elem[0]))
	#coord.append(float(line_elem[1]))
        #coord.append(float(line_elem[2]))
        coord.append(float(line_elem[0])*1.1) # Try to make initial coordinate worse
        coord.append(float(line_elem[1])*1.1)
        coord.append(float(line_elem[2])*1.1)
file_coord_read.close()


coord = numpy.array(coord)/bohr
coord_atom_init = coord.reshape((num_atom,3))
coord_com = numpy.mean(coord_atom_init,0)
coord_atom_init -= coord_com
coord = coord_atom_init.reshape((1,num_atom*3))[0]

# This is for test only
coord_bm = []
file_coord_bm = open("coord-bm",'r')
for line in file_coord_bm:
    line_elem = line.split()
    if len(line_elem)==3:
        coord_bm_atom = [float(line_elem[0]),float(line_elem[1]),float(line_elem[2])]
        coord_bm.append(coord_bm_atom)
coord_bm = numpy.array(coord_bm)/bohr
coord_bm -= numpy.mean(coord_bm,0)
file_coord_bm.close()
# Finish reading benchmark results

if force_pkg=="piny":
    # Get the initial force, you need to prepare an initial configuration to active the code you prefered
    # Let's lunch a force calculation now
    force = piny.lunch(coord,force_calc_util)
force_test = stogd.add_noise(force,num_atom)

# While testing, we output the deterministic force, we shall output the stochastic ones for real applications
file_force_traj.write("Step 0\n")
file_coord_traj.write("Step 0\n")
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

# zip the parameters for delivery
opt_param_pack = [coord,force_test,force_pkg,step_size,num_atom,file_force_traj,file_coord_traj,force_calc_util,max_shift_conv,mean_shift_conv,max_step,coord_bm]

step = 1
stop_flag = 0 # This one controls whether to stop the optimization
if opt_type=="gradient":
    stogd.sto_gd(opt_param_pack)
if opt_type=="gradient-direct":
    stogd.sto_gd_direct(opt_param_pack,num_force_store)
if opt_type=="gradient-mean":
    stogd.sto_gd_mean(opt_param_pack,num_force_store)
      
file_coord_traj.close()
file_force_traj.close()











