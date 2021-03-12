from .mechanics import *

def inverse_mechanics(self, x, v_star, envs, mnps, mode):
    if mode is None:
        print('mode cannot be None for RRTKino_w_modes class')
        raise

    # mnps = [(np.array(m.p), np.array(m.n), m.d) for m in mnps]
    # envs = [(np.array(m.p), np.array(m.n), m.d) for m in envs]
    v = qp_inv_mechanics_2d(np.array(v_star), np.array(x), mnps, envs ,mode, self.world, self.mnp_mu, self.env_mu, self.mnp_fn_max)
    return v