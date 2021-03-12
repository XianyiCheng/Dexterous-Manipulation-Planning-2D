import os
os.environ['PREHENSILE_MANIPULATION_PATH'] = '/home/xianyi/libraries/prehensile_manipulation'

import wrenchStampingLib as ws
import matlab.engine
from .mechanics import *
import numpy as np

def modes_to_int(modes):
    modes = np.array(modes)
    modes_int = np.zeros(modes.shape, dtype=int)
    modes_int[modes == CONTACT_MODE.LIFT_OFF] = 0
    modes_int[modes == CONTACT_MODE.FOLLOWING] = 1
    modes_int[modes == CONTACT_MODE.STICKING] = 1
    modes_int[modes == CONTACT_MODE.SLIDING_RIGHT] = 2
    modes_int[modes == CONTACT_MODE.SLIDING_LEFT] = 3
    return modes_int

class StabilityMarginSolver_None():
    def __init__(self):
        pass

    def preprocess(self, x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes, obj_weight, mnp_fn_max, kCharacteristicLength = 0.15):

        return 0.0


    def stability_margin(self, preprocess_params, vo, mode):

        return 0.0



class StabilityMarginSolver():
    def __init__(self):
        self.print_level=0 # 0: minimal screen outputs
        '''
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('/home/xianyi/Documents/MATLAB/tbxmanager')
        #self.eng.startup(nargout=0)
        root = '/home/xianyi/libraries/prehensile_manipulation/matlab'
        self.eng.addpath(root)
        self.eng.addpath(root + '/utilities')
        self.eng.addpath(root + '/matlablibrary')
        self.eng.addpath(root + '/matlablibrary/Math/motion')
        self.eng.addpath(root + '/matlablibrary/Math/geometry')
        self.eng.addpath(root + '/matlablibrary/Math/array')
        self.eng.addpath(root + '/matlablibrary/Mesh/cone')
        self.eng.addpath('/home/xianyi/libraries/contact_mode_enumeration_2d')
        #self.eng.supress_warning(nargout=0)
        '''

    def compute_stablity_margin(self, env_mu, mnp_mu, envs, mnps, mode, obj_weight, mnp_fn_max, dist_weight):
        '''
        #geometry stability margin
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        dist_weight = dist_weight**0.5
        kCharacteristicLength = dist_weight

        CP_W_e = []
        CN_W_e = []
        for c in envs:
            CP_W_e.append(list(c.p))
            CN_W_e.append(list(c.n))
        CP_W_e = matlab.double(np.array(CP_W_e).T.tolist())
        CN_W_e = matlab.double(np.array(CN_W_e).T.tolist())

        CP_H_h = []
        CN_H_h = []
        for c in mnps:
            CP_H_h.append(list(c.p))
            CN_H_h.append(list(c.n))

        mode_number = []
        for m in mode:
            # 0:separation 1:fixed 2: right sliding 3: left sliding
            if m == CONTACT_MODE.LIFT_OFF:
                mode_number.append(0)
            elif m == CONTACT_MODE.FOLLOWING:
                mode_number.append(1)
            elif m == CONTACT_MODE.STICKING:
                mode_number.append(1)
            elif m == CONTACT_MODE.SLIDING_RIGHT:
                mode_number.append(2)
            elif m == CONTACT_MODE.SLIDING_LEFT:
                mode_number.append(3)

        CP_H_h = matlab.double(np.array(CP_H_h).T.tolist())
        CN_H_h = matlab.double(np.array(CN_H_h).T.tolist())
        CP_W_G = matlab.double([[0],[0]])
        R_WH = matlab.double([[1,0],[0,1]])
        p_WH = matlab.double([[0],[0]])
        h_mode = matlab.int8(mode_number[0:len(mnps)])
        e_mode = matlab.int8(mode_number[len(mnps):])

        score = self.eng.tryStabilityMargin(kFrictionE, kFrictionH, CP_W_e, CN_W_e, CP_H_h, CN_H_h, CP_W_G, R_WH,
                                                     p_WH, e_mode, h_mode, float(obj_weight), float(mnp_fn_max), kCharacteristicLength)
        '''
        score = 0
        return score

    def preprocess(self, x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes, obj_weight, mnp_fn_max, kCharacteristicLength = 0.15):
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        kContactForce = mnp_fn_max

        e_modes = modes_to_int(e_modes).astype('int32')
        h_modes = modes_to_int(h_modes).astype('int32')

        # Make contact info.
        Ad_gcos, depths, mus = contact_info(mnps, envs, mnp_mu, env_mu)
        N_e = []
        T_e = []
        N_h = []
        T_h = []
        eCone_allFix = []
        hCone_allFix = []
        n = np.array([[0.0], [1.0], [0.0]])
        D = np.array([[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        n_c = len(Ad_gcos)
        for i in range(n_c):
            n_c = np.dot(Ad_gcos[i].T, n).T # 1x3
            t_c = np.dot(Ad_gcos[i].T, D).T# 2x3

            if i < len(mnps):
                f_c = n_c + t_c * kFrictionH
                f_c = f_c/np.linalg.norm(f_c[0])
                hCone_allFix.append(-f_c)
                N_h.append(-n_c[0])
                T_h.append(-t_c[0])
            else:
                f_c = n_c + t_c * kFrictionE
                f_c = f_c / np.linalg.norm(f_c[0])
                eCone_allFix.append(f_c)
                N_e.append(n_c[0])
                T_e.append(t_c[0])

        J_e = np.vstack((N_e, T_e))
        J_h = np.vstack((N_h, T_h))
        eCone_allFix = np.vstack(eCone_allFix)
        hCone_allFix = np.vstack(hCone_allFix)

        # Add gravity.
        f_g = obj_weight*np.array([[0.0], [-1.0], [0.0]])
        g_ow = np.linalg.inv(twist_to_transform(x))
        F_G = np.dot(g_ow, f_g)


        preprocess_params = (J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                             kContactForce, kFrictionE, kFrictionH,
                             kCharacteristicLength, e_modes, h_modes)

        return preprocess_params


    def stability_margin(self, preprocess_params, vo, mode):
        vo[abs(vo)<1e-5]=0
        J_e, J_h, eCone_allFix, hCone_allFix, \
        F_G,kContactForce, kFrictionE, kFrictionH,\
        kCharacteristicLength, e_modes, h_modes = preprocess_params

        mode = modes_to_int(mode).astype('int32')

        h_mode_goal = mode[0:h_modes.shape[1]].reshape(-1,1)
        e_mode_goal = mode[h_modes.shape[1]:].reshape(-1,1)

        G = np.zeros((3,6))
        G[0:3,0:3] = np.identity(3)
        b_G = np.array(vo).reshape(-1,1)

        stability_margin = ws.wrenchSpaceAnalysis_2d(J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                                                     kContactForce, kFrictionE, kFrictionH, kCharacteristicLength,
                                                     G, b_G, e_modes, h_modes, e_mode_goal, h_mode_goal, self.print_level)

        return stability_margin

    def test2d(self):

        kW = 0.0435  # object width
        kH = 0.0435  # object height
        env_mu = 0.3
        mnp_mu = 0.8

        h_modes = np.array([[CONTACT_MODE.STICKING, CONTACT_MODE.STICKING],
                            [CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.SLIDING_RIGHT],
                            [CONTACT_MODE.SLIDING_LEFT, CONTACT_MODE.SLIDING_LEFT],
                            [CONTACT_MODE.STICKING, CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.STICKING],
                            [CONTACT_MODE.SLIDING_LEFT, CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.SLIDING_RIGHT],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.SLIDING_LEFT]])

        x = (0, 2.2, 0)
        # mnps = [Contact((kW/2, kH/2),(0,-1),0),Contact((-kW/2, kH/2),(0,-1),0)]
        mnps = [Contact((0.2097357615814568, 0.2), (0, -1), 0), Contact((-0.9389810887084302, 0.2), (0, -1), 0)]
        envs = [Contact((kW / 2, -kH / 2), (0, 1), 0), Contact((-kW / 2, -kH / 2), (0, 1), 0)]
        envs = [Contact((-1.0, -0.20000000000000018), (0, 1), 0), Contact((1.0, -0.20000000000000018), (0, 1), 0)]
        mode = [CONTACT_MODE.FOLLOWING, CONTACT_MODE.FOLLOWING, CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.SLIDING_RIGHT]
        e_modes = np.array(get_contact_modes([], envs))
        e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

        object_weight = 10
        mnp_fn_max = 15
        vel = [1, 0, 0]
        v = qp_inv_mechanics_2d(np.array(vel), np.array(x), mnps, envs, mode, 'vert', mnp_mu, env_mu, mnp_fn_max)

        preprocess = self.preprocess(x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes,
                                         object_weight, mnp_fn_max)

        stability_margin_score = self.stability_margin(preprocess, v, mode)
        print(stability_margin_score)
        return stability_margin_score, preprocess, v, mode

    def hvfc_params_path(self, paths):
        path, velocity_path, mnp_path, env_path, mode_path = paths
        n = len(path)
        params_path = []
        for i in range(n):
            x = path[i]
            v = velocity_path[i]
            mode = mode_path[i]
            mnps = mnp_path[i]
            envs = env_path[i]

            mode = modes_to_int(mode).astype('int32')

            #h_mode_goal = mode[0:len(mnps)].reshape(-1, 1)
            e_mode_goal = mode[len(mnps):].reshape(-1, 1)
            e_modes = np.array(get_contact_modes([], envs))
            e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

            params_path.append((Je, Jh, eCone_allFix, hCone_allFix, F_G, G, b_G, e_modes, e_mode_goal))

        return params_path


class StabilityMarginSolver_matlab():
    def __init__(self):
        self.print_level=0 # 0: minimal screen outputs

        self.eng = matlab.engine.start_matlab()
        # self.eng.addpath('/home/xianyi/Documents/MATLAB/tbxmanager')
        #self.eng.startup(nargout=0)
        root = '/home/xianyi/libraries/prehensile_manipulation/matlab'
        self.eng.addpath(root)
        self.eng.addpath(root + '/utilities')
        self.eng.addpath(root + '/matlablibrary')
        self.eng.addpath(root + '/matlablibrary/Math/motion')
        self.eng.addpath(root + '/matlablibrary/Math/geometry')
        self.eng.addpath(root + '/matlablibrary/Math/array')
        self.eng.addpath(root + '/matlablibrary/Mesh/cone')
        #self.eng.addpath('/home/xianyi/libraries/contact_mode_enumeration_2d')
        # self.eng.supress_warning(nargout=0)
        a = -np.pi/6
        self.R_WH = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])


    def compute_stability_margin(self, x, v, env_mu, mnp_mu, envs, mnps, mode,
                                 e_modes, h_modes, obj_weight, mnp_fn_max, kCharacteristicLength=0.15):
        if sum(np.array(mode)==CONTACT_MODE.LIFT_OFF) == 2\
                and sum(np.array(mode)==CONTACT_MODE.SLIDING_LEFT)==2:
            print('debug')
        if x[2] > 2 and sum(np.array(mode)==CONTACT_MODE.SLIDING_LEFT) == 2:
            print('debug')
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        kContactForce = mnp_fn_max
        kObjWeight = obj_weight

        e_modes = modes_to_int(e_modes).astype('int32')
        h_modes = modes_to_int(h_modes).astype('int32')

        # Make contact info.
        Ad_gcos, depths, mus = contact_info(mnps, envs, mnp_mu, env_mu)

        g_wo = twist_to_transform(x)
        R_WO = g_wo[0:2, 0:2]
        p_WO = g_wo[0:2, -1].reshape(-1, 1)
        CP_W_G = p_WO

        p_O_e = np.array([c.p for c in envs]).T
        n_O_e = np.array([c.n for c in envs]).T
        p_W_e = np.dot(R_WO, p_O_e) + p_WO
        n_W_e = np.dot(R_WO, n_O_e)

        p_OH = np.array(mnps[0].p).reshape(-1, 1)
        R_OH = np.dot(R_WO.T, self.R_WH)
        goh = np.eye(3)
        goh[0:2, 0:2] = R_OH
        goh[0:2, -1] = p_OH.reshape(-1)
        Adgoh = adjoint(goh)

        p_H_h = np.zeros((2, 1))
        # n_H_h = np.array([[0],[-1]])
        R_OM = Ad_gcos[0][0:2, 0:2].T
        R_HM = np.dot(R_OH.T, R_OM)
        n_H_h = np.dot(R_HM, np.array([[0], [1]]))
        # R_WH = np.dot(R_WO, R_OH)

        # p_WH = p_WO + np.dot(R_WO,np.array(mnps[0].p))

        kNumSlidingPlanes = 1  # for 2D planar problems
        jacs = self.eng.preProcessing(matlab.double([kFrictionE]),
                                      matlab.double([kFrictionH]),
                                      matlab.double([kNumSlidingPlanes]),
                                      matlab.double([kObjWeight]),
                                      matlab.double(p_O_e.tolist()),
                                      matlab.double(n_O_e.tolist()),
                                      matlab.double(p_H_h.tolist()),
                                      matlab.double(n_H_h.tolist()),
                                      matlab.double(R_OH.tolist()),
                                      matlab.double(p_OH.tolist()),
                                      matlab.double(CP_W_G.tolist()), nargout=9)

        N_e = np.asarray(jacs[0])
        T_e = np.asarray(jacs[1])
        N_h = np.asarray(jacs[2])
        T_h = np.asarray(jacs[3])
        eCone_allFix = np.asarray(jacs[4])
        # eTCone_allFix = np.asarray(jacs[5])
        hCone_allFix = np.asarray(jacs[6])
        # hTCone_allFix = np.asarray(jacs[7])
        # F_G = np.asarray(jacs[8])
        # Add gravity.
        F_O = np.zeros(3)
        f_o = obj_weight*np.array([0.0,-1.0])
        g_ow = np.linalg.inv(twist_to_transform(x))
        F_O[0:2] = np.dot(g_ow[0:2,0:2], f_o)
        F_G = np.dot(Adgoh.T, F_O)
        F_G = F_G.reshape(-1,1)

        J_e = np.vstack((N_e, T_e))
        J_h = np.vstack((N_h, T_h))

        mode = modes_to_int(mode).astype('int32')

        h_mode_goal = mode[0:h_modes.shape[1]].reshape(-1, 1)
        e_mode_goal = mode[h_modes.shape[1]:].reshape(-1, 1)

        Adgwo = adjoint(g_wo)
        vs_w = np.dot(Adgwo, v)
        Adghw = adjoint(np.linalg.inv(np.dot(g_wo, goh)))
        vs_h = np.dot(Adghw, vs_w)

        vs_h[abs(vs_h) < 1e-5] = 0

        G = np.zeros((4,6))
        G[0:3, 0:3] = np.identity(3)
        G[3,5] = 1
        b_G = np.array(vs_h)
        idx = np.argmax(abs(b_G[0:2]))
        b_G = np.array((b_G[idx],0)).reshape(-1, 1)
        G = G[[idx,3]]
        G = G.reshape(2,-1)

        result = ws.wrenchSpaceAnalysis_2d(J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                                                     kContactForce, kFrictionE, kFrictionH, kCharacteristicLength,
                                                     G, b_G, e_modes, h_modes, e_mode_goal, h_mode_goal)

        return np.hstack((result, p_OH.reshape(-1)))



    def save_hvfc_params_path(self, paths, filename='hfvc_traj.csv'):
        env_mu = 0.3
        mnp_mu = 0.8
        obj_weight = 10
        mnp_fn_max = 100
        h_modes = np.array([[CONTACT_MODE.STICKING],[CONTACT_MODE.LIFT_OFF],[CONTACT_MODE.SLIDING_RIGHT]])

        path, velocity_path, mnp_path, env_path, mode_path = paths
        n = len(path)
        params_path = []
        for i in range(n):
            x = path[i]
            v = velocity_path[i]
            if np.linalg.norm(v) < 1e-4:
                continue
            mode = mode_path[i]
            mnps = mnp_path[i]
            envs = env_path[i]
            # mode = modes_to_int(mode).astype('int32')

            e_modes = np.array(get_contact_modes([], envs))
            e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

            params = self.compute_stability_margin(x, v, env_mu, mnp_mu, envs, mnps, mode,
                                 e_modes, h_modes, obj_weight, mnp_fn_max)


            params_path.append(params)
        params_path = np.array(params_path)
        np.savetxt(filename, params_path, delimiter=',')
        print(filename+' saved!')

        return