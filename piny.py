import sys
import math
import os
import shutil
import numpy

def create_initial(coord,force_calc_util):
    # coord in unit of Bohr
    bohr = 0.529177
    num_atom = force_calc_util[0]
    working_dir = force_calc_util[1]
    file_temp = open(working_dir+"/template.init",'r')
    file_init = open(working_dir+"/anthracene-gas.init",'w')
    count = 0
    for line in file_temp:
	line_elem = line.split()
	if len(line_elem)==5:
	    w_str = str(coord[count*3]*bohr)+' '+str(coord[count*3+1]*bohr)+' '+str(coord[count*3+2]*bohr)+' '+line_elem[3]+' '+line_elem[4]+'\n'
	    file_init.write(w_str)
	    count += 1
	else:
	    file_init.write(line)
    file_temp.close()
    file_init.close()
    

def lunch(coord,force_calc_util):
    num_atom = force_calc_util[0]
    working_dir = force_calc_util[1]
    binary_command = force_calc_util[2]
    force_file_name = force_calc_util[3]
    # create initial coordinate w.r.t. new configuration
    create_initial(coord,force_calc_util)
    #sys.exit(0)
    # Run package to generate force
    os.system(binary_command)
    force = numpy.array(numpy.loadtxt(force_file_name))
    force = force.reshape((1,num_atom*3))[0]
    # delete temp files for next iteration
    os.system("rm "+working_dir+"/out-*")
    #sys.exit(0)
    return force

