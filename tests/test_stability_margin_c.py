from numpy import *
import wrenchStampingLib as ws
from kinorrt.mechanics.stability_margin import *
from kinorrt.rrt import RRTManipulationStability

# params = (array([[ 0. ,  1. , -1. ],
#        [ 0. ,  1. ,  1. ],
#        [ 1. , -0. ,  0.2],
#        [ 1. , -0. ,  0.2]]), array([[ 0.    , -1.    , -0.2097],
#        [ 0.    , -1.    ,  0.939 ],
#        [-1.    ,  0.    ,  0.2   ],
#        [-1.    ,  0.    ,  0.2   ]]), array([[ 0.2135,  0.7118, -0.6691],
#        [-0.2135,  0.7118, -0.7545],
#        [ 0.2016,  0.6721,  0.7125],
#        [-0.2016,  0.6721,  0.6318]]), array([[-0.6242, -0.7803, -0.0388],
#        [ 0.6242, -0.7803, -0.2885],
#        [-0.4741, -0.5926,  0.6512],
#        [ 0.4741, -0.5926,  0.4616]]), array([[ -0.],
#        [-10.],
#        [  0.]]), 6, 0.3, 0.8, 0.15, array([[1., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0.],
#        [0., 0., 1., 0., 0., 0.]]), array([[-0.14  ],
#        [ 0.7001],
#        [ 0.7001]]), array([[2, 2],
#        [3, 3],
#        [1, 1],
#        [1, 0],
#        [2, 0],
#        [3, 0],
#        [0, 1],
#        [0, 2],
#        [0, 3]], dtype=int32), array([[1, 1],
#        [2, 2],
#        [3, 3],
#        [1, 0],
#        [0, 1],
#        [3, 0],
#        [2, 0],
#        [0, 2],
#        [0, 3]], dtype=int32), array([[1],
#        [0]], dtype=int32), array([[1],
#        [1]], dtype=int32), 0)
#
# stability_margin = ws.wrenchSpaceAnalysis_2d(*params)
env_mu = 0.3
mnp_mu = 0.8
object_weight = 10
mnp_fn_max = 100


stability_solver = StabilityMarginSolver()
h_modes = np.array([CONTACT_MODE.LIFT_OFF, CONTACT_MODE.STICKING,
                    CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.SLIDING_LEFT]).reshape(-1, 1)
smsolver = StabilityMarginSolver()


x = [0,0,0]
envs = [Contact((-0.5,0.2),(1,0),0),Contact((-0.5,-0.2),(1,0),0), Contact((-0.5,-0.2),(0,1),0),Contact((0.5,-0.2),(0,1),0)]
mnps = [Contact((0.5,0),(-1,0),0)]
mode = [CONTACT_MODE.FOLLOWING,CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.LIFT_OFF, CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.LIFT_OFF]
v_star = [0,0,1]

'''
x = [0,0,-np.pi/8]
envs = [Contact((-0.2,0),(1,0),0)]
mnps = [Contact((0.2,0),(-1,0),0)]
mode = [CONTACT_MODE.FOLLOWING,CONTACT_MODE.STICKING]
v_star = [0,0,1]
'''
'''
x = [0,0,0]
envs = [Contact((-0.5,-0.2),(0,1),0), Contact((0.5,-0.2),(0,1),0)]
mnps = [Contact((0.5,0),(-1,0),0)]
mode = [CONTACT_MODE.FOLLOWING,CONTACT_MODE.SLIDING_LEFT, CONTACT_MODE.SLIDING_LEFT]
v_star = [-1,0,0]
'''



vel = qp_inv_mechanics_2d(np.array(v_star), np.array(x), mnps, envs, mode, 'vert', mnp_mu, env_mu, mnp_fn_max)

e_modes = np.array(get_contact_modes([], envs))
e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]
preprocess = smsolver.preprocess(x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes,
                                      object_weight, mnp_fn_max)
# vel_ = self.inverse_mechanics(x_near, vel, envs, mnps, mode)
stability_margin_score = smsolver.stability_margin(preprocess, vel, mode)

print(stability_margin_score)
