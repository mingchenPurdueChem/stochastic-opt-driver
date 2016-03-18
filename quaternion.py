import sys
import math
import numpy
from numpy import linalg

def calc_quat(coord1,coord2):
    natom = len(coord1)
    com1 = numpy.mean(coord1,0)
    coord1 -= com1
    com2 = numpy.mean(coord2,0)
    coord2 -= com2

    cormat = numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(natom):
	cormat += numpy.outer(coord1[i],coord2[i])

    #print cormat
    Fmat = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    Fmat[0][0] = cormat[0][0]+cormat[1][1]+cormat[2][2]
    Fmat[0][1] = cormat[1][2]-cormat[2][1]
    Fmat[0][2] = cormat[2][0]-cormat[0][2]
    Fmat[0][3] = cormat[0][1]-cormat[1][0]
    Fmat[1][1] = cormat[0][0]-cormat[1][1]-cormat[2][2]
    Fmat[1][2] = cormat[0][1]+cormat[1][0]
    Fmat[1][3] = cormat[0][2]+cormat[2][0]
    Fmat[2][2] = -cormat[0][0]+cormat[1][1]-cormat[2][2]
    Fmat[2][3] = cormat[1][2]+cormat[2][1]
    Fmat[3][3] = -cormat[0][0]-cormat[1][1]+cormat[2][2]

    for i in range(4):
	for j in range(i):
	    Fmat[i][j] = Fmat[j][i]

    Fmat = numpy.array(Fmat)
    #print Fmat
    #print linalg.eigh(Fmat)

    (eigval,eigvec) = linalg.eigh(Fmat)
    #print eigvec[3]
    return [eigval[3],eigvec[:,3]]


def debug(coord1,coord2):
    natom = len(coord1)
    com1 = numpy.mean(coord1,0)
    coord1 -= com1
    com2 = numpy.mean(coord2,0)
    coord2 -= com2

    cormat = numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in range(natom):
        cormat += numpy.outer(coord1[i],coord2[i])

    #print cormat
    Fmat = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
    Fmat[0][0] = cormat[0][0]+cormat[1][1]+cormat[2][2]
    Fmat[0][1] = cormat[1][2]-cormat[2][1]
    Fmat[0][2] = cormat[2][0]-cormat[0][2]
    Fmat[0][3] = cormat[0][1]-cormat[1][0]
    Fmat[1][1] = cormat[0][0]-cormat[1][1]-cormat[2][2]
    Fmat[1][2] = cormat[0][1]+cormat[1][0]
    Fmat[1][3] = cormat[0][2]+cormat[2][0]
    Fmat[2][2] = -cormat[0][0]+cormat[1][1]-cormat[2][2]
    Fmat[2][3] = cormat[1][2]+cormat[2][1]
    Fmat[3][3] = -cormat[0][0]-cormat[1][1]+cormat[2][2]

    for i in range(4):
        for j in range(i):
            Fmat[i][j] = Fmat[j][i]

    Fmat = numpy.array(Fmat)
    #print Fmat
    #print linalg.eigh(Fmat)

    (eigval,eigvec) = linalg.eigh(Fmat)
    #print eigvec[3]
    
    diag_error = numpy.dot(Fmat,eigvec[:,3])-eigval[3]*eigvec[:,3] 
    print diag_error

    quat = eigvec[:,3]
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

    coord_rot = numpy.dot(coord1,rot_mat)
    print coord1
    print coord2
    print coord_rot
    print linalg.norm(coord1-coord2)
    print linalg.norm(coord_rot-coord2)





