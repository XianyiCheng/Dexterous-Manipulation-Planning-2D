from kinorrt.mechanics.mechanics import *
from kinorrt.mechanics.stability_margin import *
#import wrenchStampingLib as ws

def test2d():
    smsolver = StabilityMarginSolver()

    kW = 0.0435 # object width
    kH = 0.0435 # object height
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

    x =(0,2.2,0)
    #mnps = [Contact((kW/2, kH/2),(0,-1),0),Contact((-kW/2, kH/2),(0,-1),0)]
    mnps = [Contact((-0.9420752952217479, 0.2),(0,-1),0),Contact((-0.8520833515311685, 0.2),(0,-1),0)]
    #envs = [Contact((kW/2, -kH/2),(0,1),0),Contact((-kW/2, -kH/2),(0,1),0)]
    envs = [Contact((-1.0, -0.20000000000000018),(0,1),0),Contact((1.0, -0.20000000000000018),(0,1),0)]
    mode = [CONTACT_MODE.FOLLOWING, CONTACT_MODE.FOLLOWING, CONTACT_MODE.SLIDING_LEFT, CONTACT_MODE.SLIDING_LEFT]
    e_modes = np.array(get_contact_modes([], envs))
    e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

    object_weight = 10
    mnp_fn_max = 15
    vel= [-1,0,0]
    v =  qp_inv_mechanics_2d(np.array(vel), np.array(x), mnps, envs, mode, 'vert', mnp_mu, env_mu,mnp_fn_max)

    preprocess = smsolver.preprocess(x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes,
                                          object_weight, mnp_fn_max)

    stability_margin_score = smsolver.stability_margin(preprocess, v, mode)
    print(stability_margin_score)

if __name__ == "__main__":
    test2d()