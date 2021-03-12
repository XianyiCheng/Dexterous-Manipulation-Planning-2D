import matlab.engine
eng = matlab.engine.connect_matlab()
root = '/home/xianyi/Dropbox/MLAB/PlanningThroughContact/ContactSeq_Planning/code/kinorrt/src/kinorrt/mechanics/prehensile_manipulation/matlab'
eng.addpath(root)
eng.addpath(root + '/utilities')
eng.addpath(root + '/matlablibrary')
eng.addpath(root + '/matlablibrary/Math/motion')
eng.addpath(root + '/matlablibrary/Math/geometry')
eng.addpath(root + '/matlablibrary/Math/array')
eng.addpath(root + '/matlablibrary/Mesh/cone')
eng.addpath('/home/xianyi/Dropbox/MLAB/ElementofGrasp/code/contact_mode_enumeration/contact_mode_enumeration_2d')
import numpy as np
from numpy import newaxis


import contact_modes as cm
import wrenchStampingLib as ws

# Parameters
kFrictionH = 0.7
kFrictionE = 0.3

kContactForce = 15
kObjWeight = 5

kCharacteristicLength = 0.15

##
## Geometrical Problem definition
##
#
#             -----------
#          h2|          | h1
#        --------------------
#        |        Y         |
#        |        ^         |
#     e2 |        | O       | e1
#    =============|---> X ===========
#
kW = 0.0435 # object width
kH = 0.0435 # object height

# list of contact points and contact normals
p_W_e = np.array(([kW/2, 0],
                  [-kW/2, 0])).T
n_W_e = np.array(([0, 1],
                  [0, 1])).T
p_H_h = np.array(([kW/2, 0],
                  [-kW/2, 0])).T
n_H_h = np.array(([0, -1],
                  [0, -1])).T

CP_W_G = np.array([0, kH/2]);
CP_W_G = CP_W_G[:, newaxis]

R_WH = np.eye(2)
p_WH = np.array(([0, kH]))
p_WH = p_WH[:, newaxis]

##
## Geometrical Pre-processing
##

kNumSlidingPlanes = 1 # for 2D planar problems
jacs = eng.preProcessing(matlab.double([kFrictionE]),
        matlab.double([kFrictionH]),
        matlab.double([kNumSlidingPlanes]),
        matlab.double([kObjWeight]),
        matlab.double(p_W_e.tolist()),
        matlab.double(n_W_e.tolist()),
        matlab.double(p_H_h.tolist()),
        matlab.double(n_H_h.tolist()),
        matlab.double(R_WH.tolist()),
        matlab.double(p_WH.tolist()),
        matlab.double(CP_W_G.tolist()), nargout=9)
# print('jacs:')
# print(np.array(jacs))

# read outputs
N_e = np.asarray(jacs[0])
T_e = np.asarray(jacs[1])
N_h = np.asarray(jacs[2])
T_h = np.asarray(jacs[3])
eCone_allFix = np.asarray(jacs[4])
eTCone_allFix = np.asarray(jacs[5])
hCone_allFix = np.asarray(jacs[6])
hTCone_allFix = np.asarray(jacs[7])
F_G = np.asarray(jacs[8])

b_e = np.zeros((N_e.shape[0], 1))
t_e = np.zeros((eTCone_allFix.shape[0], 1))
b_h = np.zeros((N_h.shape[0], 1))
t_h = np.zeros((hTCone_allFix.shape[0], 1))

J_e = np.vstack((N_e, T_e))
J_h = np.vstack((N_h, T_h))

# [e_modes, h_modes] = sharedGraspModeEnumeration(CP_W_e, CN_W_e, CP_H_h, CN_H_h);
modes = eng.sharedGraspModeEnumeration(matlab.double(p_W_e.tolist()),
        matlab.double(n_W_e.tolist()),
        matlab.double(p_H_h.tolist()),
        matlab.double(n_H_h.tolist()), nargout = 2);
e_modes = np.asarray(modes[0]).astype('int32').T
h_modes = np.asarray(modes[1]).astype('int32').T

##
## Goal
##

# Palm Pivot
G = np.array([0., 0., 1., 0., 0., 0.]);
G = G[newaxis, :]
b_G = np.array([[0.1]]);
e_mode_goal = np.array([[0, 1]]).astype('int32').T # sf
h_mode_goal = np.array([[1, 1]]).astype('int32').T # ff

print_level = 0; # 0: minimal screen outputs
stability_margin = ws.wrenchSpaceAnalysis_2d(J_e, J_h, eCone_allFix, hCone_allFix, F_G,
    kContactForce, kFrictionE, kFrictionH, kCharacteristicLength,
    G, b_G, e_modes, h_modes, e_mode_goal, h_mode_goal, print_level)

print ('stability_margin = ')
print (stability_margin)
