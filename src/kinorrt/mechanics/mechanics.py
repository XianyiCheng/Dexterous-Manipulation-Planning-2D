import numpy as np
import quadprog
#from scipy.optimize import linprog
from qpsolvers import solve_qp

from enum import Enum
from contact_modes.modes_3d import enum_sliding_sticking_3d_proj
from contact_modes.constraints import build_tangent_velocity_constraints_2d, build_normal_velocity_constraints_2d
#import gurobipy as gp

debug = False

class Contact(object):
    def __init__(self, position, normal, depth):
        self.p = tuple(position)
        self.n = tuple(normal)
        self.d = depth
    def is_same(self, c):
        if np.dot(self.n,c.n) < 1e-3:
            return False
        else:
            return True

class CONTACT_MODE(Enum):
    STICKING      = 0
    SLIDING_LEFT  = 1
    SLIDING_RIGHT = 2
    LIFT_OFF      = 3
    FOLLOWING     = 4


class Sparse:
    def __init__(self):
        self.data = dict()
        self.num_cols = 0
        self.num_rows = 0

    def size(self):
        return self.num_rows, self.num_cols

    def resize(self, rows, cols):
        if rows > 0:
            self.num_rows = rows
        if cols > 0:
            self.num_cols = cols

    def clear(self):
        self.num_cols = 0
        self.num_rows = 0
        self.data.clear()

    def add(self, x, i, j):
        if i >= self.num_rows:
            self.num_rows = i + 1
        if j >= self.num_cols:
            self.num_cols = j + 1
        self.data[(i,j)] = x

    def append(self, X, J):
        i = self.num_rows
        for x, j in zip(X, J):
            self.add(x, i, j)

    def numpy(self):
        full = np.zeros((self.num_rows, self.num_cols))
        for ind, val in self.data.items():
            full[ind[0], ind[1]] = val
        return full

def v_hat(v):
    V = np.zeros([3,3])
    V[0:2, 0:2] = np.array([[0, -v[2]], [v[2], 0]])
    V[0:2, 2] = v[0:2].reshape(-1)
    return V

def config2trans(q):
    q = np.array(q).flatten()
    a = q[2]
    g = np.identity(3)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a), np.cos(a)]])
    g[0:2,0:2] = R
    g[0:2,-1] = q[0:2]
    return g

def twist_to_transform(x):
    x = np.array(x).reshape((3,1))
    c = np.cos(x[2,0])
    s = np.sin(x[2,0])

    g = np.eye(3)
    g[0:2,0:2] = np.array([[c, -s], [s, c]])
    g[0:2,2] = x[0:2,0]
    return g

def get_contact_frame(contact_point, normal):
    contact_point = np.array(contact_point).reshape((2,1))
    normal = np.array(normal).reshape((2,1))
    cw90 = np.array([[0.0, 1.0],[-1.0, 0.0]])

    g = np.eye(3)
    g[0:2,0,None] = np.dot(cw90, normal)
    g[0:2,1,None] = normal
    g[0:2,2,None] = contact_point
    return g

def get_contact_forces(contacts, f):
    n_c = len(contacts)
    forces = np.zeros((2, n_c))
    for i in range(n_c):
        g = get_contact_frame(contacts[i].p, contacts[i].n)
        f_n = f[3*i+0]
        f_p = f[3*i+1]
        f_m = f[3*i+2]
        x = (f_p - f_m) * g[0:2,0]
        y = f_n * g[0:2,1]
        forces[:,i] = x + y
    return forces

def adjoint(g):
    Ad = np.eye(3)
    Ad[0:2,0:2] = g[0:2,0:2]
    Ad[0,2] =  g[1,2]
    Ad[1,2] = -g[0,2]
    return Ad

def adjoint_inverse(g):
    return np.linalg.inv(adjoint(g))

def add_contact_constraint_2d(Ad_gco, depth, mu, mode, c_id, G, h, A, b):
    """Adds the specified mode of a contact complementarity constraint to the
    constraint matrices. This function assumes that the object twist v_o ∈ R³,
    each contact force is parameterized by fₙ, fₜ⁺, fₜ⁻ (all ≥ 0), and that the
    incident geometries are static.
    Arguments:
        Ad_gco {np.ndarray} -- adjoint of transform from object to contact frame
        depth {float} -- penetration depth
        mu {float} -- coefficient of friction
        mode {CONTACT_MODE} -- contact mode (see enum)
        c_id {int} -- contact index
        G {Sparse} -- Gx ≥ h
        h {Sparse} --
        A {Sparse} -- Ax = b
        b {Sparse} --
    """

    # Offset to contact force variables.
    f_id = 3 + 3 * c_id

    if mode == CONTACT_MODE.LIFT_OFF:
        # Contact forces = 0.
        A.append([1], [f_id])
        A.append([1], [f_id+1])
        A.append([1], [f_id+2])
        b.append([0], [0])
        b.append([0], [0])
        b.append([0], [0])

        # Normal velocity ≥ 0.
        n = np.array([[0.0],[1.0],[0.0]])
        nAd_gco = np.dot(n.T, Ad_gco).flatten()
        G.append(nAd_gco.tolist(), [0, 1, 2])
        h.append([-depth], [0])

    elif mode == CONTACT_MODE.FOLLOWING:
        # Contact normal force ≥ 0.
        G.append([1], [f_id])
        h.append([0], [0])

        # Contact tangential force ≥ 0.
        G.append([1], [f_id+1])
        G.append([1], [f_id+2])
        h.append([0], [0])
        h.append([0], [0])

        # Coloumb friction, μfₙ-(fₜ⁺-fₜ⁻) ≥ 0.
        G.append([mu, -1, -1], [f_id, f_id+1, f_id+2])
        h.append([0], [0])

    else:
        # Contact normal force ≥ 0.
        G.append([1], [f_id])
        h.append([0], [0])

        # Normal velocity = 0.
        n = np.array([[0.0],[1.0],[0.0]])
        nAd_gco = np.dot(n.T, Ad_gco).flatten()
        A.append(nAd_gco.tolist(), [0, 1, 2])
        b.append([-depth], [0])

        # Tangential velocity matrix (≶ ?)
        D = np.array([[1.0],[0.0],[0.0]])
        DAd_gco = np.dot(D.T, Ad_gco).flatten()

        if mode == CONTACT_MODE.STICKING:
            # Contact tangential force ≥ 0.
            G.append([1], [f_id+1])
            G.append([1], [f_id+2])
            h.append([0], [0])
            h.append([0], [0])

            # Coloumb friction, μfₙ-(fₜ⁺-fₜ⁻) ≥ 0.
            G.append([mu, -1, -1], [f_id, f_id+1, f_id+2])
            h.append([0], [0])

            # Tangential velocity = 0.
            A.append(DAd_gco.tolist(), [0, 1, 2])
            b.append([0], [0])

        if mode == CONTACT_MODE.SLIDING_LEFT:
            # Contact tangential force, μfₙ=fₜ⁺ and fₜ⁻=0
            A.append([mu, -1], [f_id, f_id+1])
            b.append([0], [0])
            A.append([1], [f_id+2])
            b.append([0], [0])

            # Tangential velocity ≤ 0.
            G.append((-DAd_gco).tolist(), [0, 1, 2])
            h.append([0], [0])

        if mode == CONTACT_MODE.SLIDING_RIGHT:
            # Contact tangential force, μfₙ=fₜ⁻ and fₜ⁺=0
            A.append([mu, -1], [f_id, f_id+2])
            b.append([0], [0])
            A.append([1], [f_id+1])
            b.append([0], [0])

            # Tangential velocity ≥ 0.
            G.append((DAd_gco).tolist(), [0, 1, 2])
            h.append([0], [0])

def add_fixed_mode_constraints(Ad_gcos, depths, mus, modes, G, h, A, b):
    num_contacts = len(modes)
    for i in range(num_contacts):
        add_contact_constraint_2d(
            Ad_gcos[i], depths[i], mus[i], modes[i], i, G, h, A, b)

def make_contact_info(mnp, env, mnp_mu, env_mu, x=None):
    if x is None:
        x = np.zeros((3,))
    g_ow = np.linalg.inv(twist_to_transform(x))
    R_ow = g_ow[0:2,0:2]
    t_ow = g_ow[0:2,2]
    Ad_gcos = []
    depths = []
    mus = []
    env_points = np.zeros((2, len(env)))
    env_normals = np.zeros((2, len(env)))
    for i in range(len(mnp)):
        p, n, d = mnp[i]
        p = np.dot(R_ow, p) + t_ow
        n = np.dot(R_ow, n)
        Ad_gcos.append(adjoint_inverse(get_contact_frame(p, n)))
        depths.append(d)
        mus.append(mnp_mu)

    if len(env) != 0:
        for i in range(len(env)):
            p = env[i].p
            n = env[i].n
            d = env[i].d
            p = np.dot(R_ow, p) + t_ow
            n = np.dot(R_ow, n)
            env_points[:,i] = p
            env_normals[:,i] = n
            Ad_gcos.append(adjoint_inverse(get_contact_frame(p, n)))
            depths.append(d)
            mus.append(env_mu)

        # enumerate env contact modes
        A, b = build_normal_velocity_constraints_2d(env_points,env_normals)
        T, t = build_tangent_velocity_constraints_2d(env_points,env_normals)
        env_modes = contact_mode_interpret(enum_sliding_sticking_3d_proj(A, b, T, t)[0])
        modes = [[CONTACT_MODE.FOLLOWING]*len(mnp) + m for m in env_modes]
    else:
        modes = [[CONTACT_MODE.FOLLOWING] * len(mnp)]

    # assume all mnp contacts are following

    return Ad_gcos, depths, mus, modes

def contact_info(mnp, env, mnp_mu, env_mu, x=None):
    if x is None:
        x = np.zeros((3,))
    g_ow = np.linalg.inv(twist_to_transform(x))
    R_ow = g_ow[0:2,0:2]
    t_ow = g_ow[0:2,2]
    Ad_gcos = []
    depths = []
    mus = []

    env_points = np.zeros((2, len(env)))
    env_normals = np.zeros((2, len(env)))
    for i in range(len(mnp)):
        p = mnp[i].p
        n = mnp[i].n
        d = mnp[i].d
        p = np.dot(R_ow, p) + t_ow
        n = np.dot(R_ow, n)
        Ad_gcos.append(adjoint_inverse(get_contact_frame(p, n)))
        depths.append(d)
        mus.append(mnp_mu)

    if len(env) != 0:
        for i in range(len(env)):
            p = env[i].p
            n = env[i].n
            d = env[i].d
            p = np.dot(R_ow, p) + t_ow
            n = np.dot(R_ow, n)
            env_points[:,i] = p
            env_normals[:,i] = n
            Ad_gcos.append(adjoint_inverse(get_contact_frame(p, n)))
            depths.append(d)
            mus.append(env_mu)


    # assume all mnp contacts are following

    return Ad_gcos, depths, mus

def get_contact_modes(mnp, env):

    env_points = np.zeros((2, len(env)))
    env_normals = np.zeros((2, len(env)))
    # for i in range(len(mnp)):
    #     p = mnp[i].p
    #     n = mnp[i].n

    if len(env) != 0:
        for i in range(len(env)):
            env_points[:,i] = env[i].p
            env_normals[:,i] = env[i].n

        # enumerate env contact modes
        A, b = build_normal_velocity_constraints_2d(env_points,env_normals)
        T, t = build_tangent_velocity_constraints_2d(env_points,env_normals)
        env_modes = contact_mode_interpret(enum_sliding_sticking_3d_proj(A, b, T, t)[0])
        modes = [[CONTACT_MODE.FOLLOWING]*len(mnp) + m for m in env_modes]
    else:
        modes = [[CONTACT_MODE.FOLLOWING] * len(mnp)]

    # assume all mnp contacts are following

    return modes

def contact_mode_interpret(contact_modes):
    modes = []
    n_pts = int(len(contact_modes[0])/2)
    for i in range(len(contact_modes)):
        cm = np.array(contact_modes[i])
        m = np.array([CONTACT_MODE.LIFT_OFF]*n_pts)
        m[cm[n_pts:] == 0] = CONTACT_MODE.STICKING
        m[cm[n_pts:] == 1] = CONTACT_MODE.SLIDING_RIGHT
        m[cm[n_pts:] == -1] = CONTACT_MODE.SLIDING_LEFT
        m[cm[0:n_pts] == 1] = CONTACT_MODE.LIFT_OFF # no matter what cm[n_pts:] is
        modes.append(list(m))
    #print(modes)
    return modes

def add_velocity_cost(v, P, q):
    v = v.reshape((3,))
    # Add the cost (x - v)ᵀ(x - v)
    P.add(1, 0, 0)
    P.add(1, 1, 1)
    P.add(1, 2, 2)
    q.add(v[0], 0, 0) # 1/2 is pre-factored into cost
    q.add(v[1], 1, 0)
    q.add(v[2], 2, 0)

def add_force_norm_cost(n_c, P, q):
    # Add a cost on force magnitude.
    for i in range(n_c):
        f_id = 3 + 3 * i
        P.add(0.001, f_id, f_id)
        P.add(0.001, f_id+1, f_id+1)
        P.add(0.005, f_id+2, f_id+2)

def add_fm_balance_vert_2d(x, Ad_gcos, mus, A, b):
    # Add contact forces.

    n = np.array([[0.0],[1.0],[0.0]])
    D = np.array([[1.0, -1.0],[0.0, 0.0],[0.0, 0.0]])
    r, _ = A.size()
    n_c = len(Ad_gcos)
    for i in range(n_c):
        N_c = np.dot(Ad_gcos[i].T, n)
        T_c = np.dot(Ad_gcos[i].T, D)
        F_c = np.concatenate((N_c, T_c), axis=1)
        f_id = 3 + 3 * i
        for ii in range(F_c.shape[0]):
            for jj in range(F_c.shape[1]):
                A.add(F_c[ii,jj], r+ii, f_id+jj)
    # Add gravity.
    f_g = np.array([[0.0],[10.0],[0.0]])
    g_ow = np.linalg.inv(twist_to_transform(x))
    f_o = np.dot(g_ow, f_g)
    for ii in range(3):
        b.add(f_o[ii], r+ii, 0)

def add_mnp_force_limits(n_m, mnp_fn_max, G, h):
    # Add normal force limits.
    for i in range(n_m):
        f_id = 3 + 3 * i
        G.append([-1], [f_id])
        h.append([-mnp_fn_max], [0])

def add_fm_balance_planar_2d(x, Ad_gcos, mus, A, b):
    # Add contact forces.
    n = np.array([[0.0], [1.0], [0.0]])
    D = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
    r, _ = A.size()
    n_c = len(Ad_gcos)
    for i in range(n_c):
        N_c = np.dot(Ad_gcos[i].T, n)
        T_c = np.dot(Ad_gcos[i].T, D)
        F_c = np.concatenate((N_c, T_c), axis=1)
        f_id = 3 + 3 * i
        for ii in range(F_c.shape[0]):
            for jj in range(F_c.shape[1]):
                A.add(F_c[ii, jj], r + ii, f_id + jj)

    # Add force motion
    inv_S = np.array([[1,0,0],[0,1,0],[0,0,5]])
    for ii in range(inv_S.shape[0]):
        for jj in range(inv_S.shape[0]):
            A.add(-inv_S[ii,jj], r+ii, jj)
    for ii in range(3):
        b.add(0,r+ii,0)
    return

def qplcp_quadprog(P, q, G, h, A, b):
    qp_G = 0.5 * (P + P.T) # make sure P is symmetric
    qp_a = q.flatten()
    qp_C = np.concatenate([A, G], axis=0).T
    qp_b = np.concatenate([b, h], axis=0).flatten()
    meq = A.shape[0]
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def qplcp(P, q, G, h, A, b):
    P = 0.5 * (P + P.T) # make sure P is symmetric
    q = -q.flatten()
    h = h.flatten()
    b = b.flatten()

    return solve_qp(P, q, -G, -h, A, b)

def gurobi_solver(P, p, G, h, A, b):
    n_var = p.shape[0]

    p = p.flatten()
    h = h.flatten()
    b = b.flatten()
    model = gp.Model('lp')

    x = model.addVars(n_var,lb = -gp.GRB.INFINITY, name = 'x')

    x_list = [x[i] for i in range(n_var)]
    model.addMConstrs(A, x_list, '=', b)
    model.addMConstrs(G, x_list, '>=', h)

    model.setMObjective(P,-2*p,0.0,xQ_L=x_list, xQ_R=x_list, xc=x_list, sense = gp.GRB.MINIMIZE)
    #model.setParam('BarHomogeneous',1)
    model.setParam('OutputFlag', 0)
    model.optimize()

    ifsuccess = model.status == 2
    if ifsuccess:
        vars = model.getVars()
        z = [vars[i].x for i in range(n_var)]
    else:
        z = None

    return z, ifsuccess

def inv_vert_2d(v, x, mnp, env, mnp_mu=0.5, env_mu=0.25, mnp_fn_max=None):

    """Solve inverse task mechanics for object motion in a 2d vertical world.
    Arguments:
        v {np.ndarray} -- target object body velocity
        x {np.ndarray} -- object pose (twist)
        mnp {list} -- list of object frame manipulator contacts [[p, n, d],...]
        env {list} -- list of object frame environment contacts [[p, n, d],...]
    Keyword Arguments:
        mnp_mu {float} -- manipulator friction (default: {0.5})
        env_mu {float} -- environment friction (default: {0.25})

    Returns:
        v_o {np.ndarray} -- closest feasible object body velocity
        qss {bool} -- quasi-statically stable: false if the system had no solutions
    """

    # Get variable sizes.
    n_m = len(mnp)
    n_c = len(mnp) + len(env)
    n_z = 3 + 3 * n_c

    v = v.flatten()

    # Make contact info.
    Ad_gcos, depths, mus, contact_modes = make_contact_info(mnp, env, mnp_mu, env_mu) #, x) for world frame instead
    contacts = mnp.copy()
    contacts.extend(env)

    # Cost matrices.
    P = Sparse()
    q = Sparse()

    # Add velocity + force norm costs, i.e. min |x-v|² + |f|²
    add_velocity_cost(v, P, q)
    add_force_norm_cost(len(mnp)+len(env), P, q)

    # Full cost matrices.
    P.resize(n_z, n_z)
    q.resize(n_z, 1)
    P = P.numpy()
    q = q.numpy()

    # Constraint matrices.
    G = Sparse()    # inequality
    h = Sparse()
    A = Sparse()    # equality
    b = Sparse()

    # Enumerate contact modes.
    min_cost = np.inf
    v_o = np.zeros((3,1))
    f_o = np.zeros((3 * n_c, 1))
    qss = 0
    for i in range(len(contact_modes)):
        mode = contact_modes[i]
        # Reset constraints.
        G.clear()
        h.clear()
        A.clear()
        b.clear()

        # Add force-moment balance.
        add_fm_balance_vert_2d(x, Ad_gcos, mus, A, b)

        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)

        # Add manipulator force limits.
        if mnp_fn_max > 0:
            add_mnp_force_limits(n_m, mnp_fn_max, G, h)

        # Resize to full matrices.
        G.resize(-1, n_z)
        A.resize(-1, n_z)

        # Solve.
        try:
            z= qplcp(P, q, G.numpy(), h.numpy(), A.numpy(), b.numpy())
            qss = 1

            if debug:
                print(z)
                print(G.numpy())
                print(A.numpy())

        except:
            pass
        else:
            z_o = z[0:3]
            cost = np.linalg.norm(v - z_o)

            if cost < min_cost:
                min_cost = cost
                v_o = z_o.reshape((3,1))
                f_o = get_contact_forces(contacts, z[3:])

    #print(min_cost)

    return v_o

def inv_planar_2d(v, x, mnp, env, mnp_mu=0.5, env_mu=0.25):
    """Solve inverse task mechanics for object motion in a 2d planar world.
    Arguments:
        v {np.ndarray} -- target object body velocity
        x {np.ndarray} -- object pose (twist)
        mnp {list} -- list of object frame manipulator contacts [[p, n, d],...]
        env {list} -- list of object frame environment contacts [[p, n, d],...]
    Keyword Arguments:
        mnp_mu {float} -- manipulator friction (default: {0.5})
        env_mu {float} -- environment friction (default: {0.25})
    """
    # Get variable sizes.

    v = v.flatten()
    n_m = len(mnp)
    n_c = len(mnp) + len(env)
    n_z = 3 + 3 * n_c

    # Make contact info.
    Ad_gcos, depths, mus, contact_modes = \
        make_contact_info(mnp, env, mnp_mu, env_mu) #, x) for world frame instead
    #contacts = mnp.copy().extend(env)
    contacts = mnp + env
    # Cost matrices.
    P = Sparse()
    q = Sparse()

    # Add velocity + force norm costs, i.e. min |x-v|² + |f|²
    add_velocity_cost(v, P, q)
    add_force_norm_cost(len(mnp)+len(env), P, q)

    # Full cost matrices.
    P.resize(n_z, n_z)
    q.resize(n_z, 1)
    P = P.numpy()
    q = q.numpy()

    # Constraint matrices.
    G = Sparse()    # inequality
    h = Sparse()
    A = Sparse()    # equality
    b = Sparse()

    # Enumerate contact modes.
    min_cost = np.inf
    v_o = np.zeros((3, 1))
    f_o = np.zeros((3, len(contacts)))
    for i in range(len(contact_modes)):
        mode = contact_modes[i]
        # Reset constraints.
        G.clear()
        h.clear()
        A.clear()
        b.clear()

        # TODO: Add force-moment balance.
        add_fm_balance_planar_2d(x, Ad_gcos, mus, A, b)

        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)

        # Resize to full matrices.
        G.resize(-1, n_z)
        A.resize(-1, n_z)


        # Solve.
        try:
            z = qplcp(P, q, G.numpy(), h.numpy(), A.numpy(), b.numpy())
            qss = 1

            if debug:
                print(z)
                print(G.numpy())
                print(A.numpy())

        except ValueError as e:
            pass
            #print('Quadprog no solution')
        else:
            z_o = z[0:3]
            cost = np.linalg.norm(v - z_o)

            if cost < min_cost:
                min_cost = cost
                v_o = z_o.reshape((3, 1))
                f_o = get_contact_forces(contacts, z[3:])


    #print(min_cost)

    return v_o

#@profile
def qp_inv_mechanics_2d(v, x, mnp, env, mode, world, mnp_mu=0.8, env_mu=0.3, mnp_fn_max=None):
    """Solve inverse task mechanics for object motion in a 2d planar world.
    Arguments:
        v {np.ndarray} -- target object body velocity
        x {np.ndarray} -- object pose (twist)
        mnp {list} -- list of object frame manipulator contacts [[p, n, d],...]
        env {list} -- list of object frame environment contacts [[p, n, d],...]
    Keyword Arguments:
        mnp_mu {float} -- manipulator friction (default: {0.5})
        env_mu {float} -- environment friction (default: {0.25})
    """
    # Get variable sizes.

    v = v.flatten()
    n_m = len(mnp)
    n_c = len(mnp) + len(env)
    n_z = 3 + 3 * n_c

    # Make contact info.
    Ad_gcos, depths, mus = contact_info(mnp, env, mnp_mu, env_mu) #, x) for world frame instead
    #contacts = mnp.copy().extend(env)
    contacts = list(mnp) + list(env)
    # Cost matrices.
    P = Sparse()
    q = Sparse()

    # Add velocity + force norm costs, i.e. min |x-v|² + |f|²
    add_velocity_cost(v, P, q)
    add_force_norm_cost(len(mnp)+len(env), P, q)

    # Full cost matrices.
    P.resize(n_z, n_z)
    q.resize(n_z, 1)
    P = P.numpy()
    q = q.numpy()

    # Constraint matrices.
    G = Sparse()    # inequality Gx>=h
    h = Sparse()
    A = Sparse()    # equality
    b = Sparse()

    # Enumerate contact modes.
    v_o = np.zeros((3, 1))
    f_o = np.zeros((3, len(contacts)))

    if world == 'planar':
        add_fm_balance_planar_2d(x, Ad_gcos, mus, A, b)
        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)
    elif world == 'vert':
        # Add force-moment balance.
        add_fm_balance_vert_2d(x, Ad_gcos, mus, A, b)
        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)
        # Add manipulator force limits.
        if mnp_fn_max is not None:
            add_mnp_force_limits(n_m, mnp_fn_max, G, h)
    else:
        print('world can only be vert or planar')
        raise

    # Resize to full matrices.
    G.resize(-1, n_z)
    A.resize(-1, n_z)


    # Solve.
    # z = qplcp_quadprog(P, q, G.numpy(), h.numpy(), A.numpy(), b.numpy())
    # print(z)
    # z = qplcp(P, q, G.numpy(), h.numpy(), A.numpy(), b.numpy())
    # print(z)
    try:
        z = qplcp(P, q, G.numpy(), h.numpy(), A.numpy(), b.numpy())
        #zg, ifsuccess = gurobi_solver(P, q, G.numpy(), h.numpy(), A.numpy(), b.numpy())
        #qss = 1

    except:
        pass
        #print('Quadprog no solution')
    else:
        z_o = z[0:3]
        cost = np.linalg.norm(v - z_o)
        v_o = z_o.reshape((3, 1))
        f_o = get_contact_forces(contacts, z[3:])

    return v_o

#@profile
def static_equilibrium(x, mnp, env, world, mnp_mu=0.8, env_mu=0.3, mnp_fn_max=None, mode = None):
    """test if there is a static equilibrium
    Arguments:
        v {np.ndarray} -- target object body velocity
        x {np.ndarray} -- object pose (twist)
        mnp {list} -- list of object frame manipulator contacts [[p, n, d],...]
        env {list} -- list of object frame environment contacts [[p, n, d],...]
    Keyword Arguments:
        mnp_mu {float} -- manipulator friction (default: {0.5})
        env_mu {float} -- environment friction (default: {0.25})
    """
    x = np.array(x)
    # Get variable sizes.
    n_m = len(mnp)
    n_c = len(mnp) + len(env)
    n_z = 3 + 3 * n_c

    if mode is None:
        mode = [CONTACT_MODE.FOLLOWING]*n_m + [CONTACT_MODE.STICKING]*len(env)

    # Make contact info.
    Ad_gcos, depths, mus = contact_info(mnp, env, mnp_mu, env_mu) #, x) for world frame instead
    depths = [0.0]*n_c
    #contacts = mnp.copy().extend(env)
    contacts = list(mnp) + list(env)

    # Constraint matrices.
    G = Sparse()    # inequality Gx >= h
    h = Sparse()
    A = Sparse()    # equality Ax = b
    b = Sparse()

    if world == 'planar':
        add_fm_balance_planar_2d(x, Ad_gcos, mus, A, b)
        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)
    elif world == 'vert':
        # Add force-moment balance.
        add_fm_balance_vert_2d(x, Ad_gcos, mus, A, b)
        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)
        # Add manipulator force limits.
        if mnp_fn_max is not None:
            add_mnp_force_limits(n_m, mnp_fn_max, G, h)
    else:
        print('world can only be vert or planar')
        raise

    # Resize to full matrices.
    G.resize(-1, n_z)
    A.resize(-1, n_z)

    # Solve.
    try:
        #gurobi_solver(np.zeros((n_z,n_z)), np.zeros(n_z), G.numpy(), h.numpy(), A.numpy(), b.numpy())
        #res = linprog(np.zeros(n_z), A_ub=-G.numpy(), b_ub= -h.numpy(), A_eq = A.numpy(), b_eq = b.numpy())
        qplcp(np.identity(n_z), np.zeros(n_z), G.numpy(), h.numpy(), A.numpy(), b.numpy())
        ifsuccess = True #res.success
    except:
        ifsuccess = False

    return ifsuccess

def velocity_project_direction(v, mnp, env, mode, mnp_mu=0.8, env_mu=0.3):
    # project velocity to its positive direction that's feasible under given contact mode
    # return: d_proj: unit vector in the positive direction (v_star_proj = v_star*d_proj * d_proj)
    """Solve inverse task mechanics for object motion in a 2d planar world.
     Arguments:
         v {np.ndarray} -- target object body velocity
         x {np.ndarray} -- object pose (twist)
         mnp {list} -- list of object frame manipulator contacts [[p, n, d],...]
         env {list} -- list of object frame environment contacts [[p, n, d],...]
     Keyword Arguments:
         mnp_mu {float} -- manipulator friction (default: {0.5})
         env_mu {float} -- environment friction (default: {0.25})
     """
    # Get variable sizes.

    v = np.array(v)
    n_m = len(mnp)
    n_c = len(mnp) + len(env)
    n_z = 3 + 3 * n_c

    # Make contact info.
    Ad_gcos, depths, mus = contact_info(mnp, env, mnp_mu, env_mu)  # , x) for world frame instead
    # contacts = mnp.copy().extend(env)
    contacts = list(mnp) + list(env)
    # Cost matrices.
    P = np.identity(3)
    q = (v).reshape(1,-1)

    # Constraint matrices.
    G = Sparse()  # inequality
    h = Sparse()
    A = Sparse()  # equality
    b = Sparse()

    add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)

    G = G.numpy()
    A = A.numpy()
    h = h.numpy()
    b = b.numpy()
    G = G[:,0:3]
    A = A[:,0:3]
    G_index = np.any(G!=0,axis=1)
    G = G[G_index,:]
    h = h[G_index]
    A_index = np.any(A!=0,axis=1)
    A = A[A_index,:]
    b = b[A_index]

    # Solve.
    try:
        z = qplcp(P, q, G, h, A, b)
        if np.linalg.norm(z) < 1e-3:
            d_proj = z
        else:
            d_proj = z / np.linalg.norm(z)
    except:
        d_proj = np.zeros(3)

    return d_proj

def contact_mode_constraints(x, mnp, env, mode, world, mnp_mu=0.8, env_mu=0.3, mnp_fn_max=None):
    """contact mode constraints object in 2d
    Arguments:
        x {np.ndarray} -- object pose (twist)
        mnp {list} -- list of object frame manipulator contacts [[p, n, d],...]
        env {list} -- list of object frame environment contacts [[p, n, d],...]
    Keyword Arguments:
        mnp_mu {float} -- manipulator friction (default: {0.5})
        env_mu {float} -- environment friction (default: {0.25})
    Return:
        Gq>=h
        Aq=b
    """
    # Get variable sizes.

    n_m = len(mnp)
    n_c = len(mnp) + len(env)
    n_z = 3 + 3 * n_c

    # Make contact info.
    Ad_gcos, depths, mus = contact_info(mnp, env, mnp_mu, env_mu) #, x) for world frame instead
    #contacts = mnp.copy().extend(env)
    contacts = list(mnp) + list(env)

    # Constraint matrices.
    G = Sparse()    # inequality
    h = Sparse()
    A = Sparse()    # equality
    b = Sparse()

    if world == 'planar':
        add_fm_balance_planar_2d(x, Ad_gcos, mus, A, b)
        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)
    elif world == 'vert':
        # Add force-moment balance.
        add_fm_balance_vert_2d(x, Ad_gcos, mus, A, b)
        # Add fixed-mode contact constraints.
        add_fixed_mode_constraints(Ad_gcos, depths, mus, mode, G, h, A, b)
        # Add manipulator force limits.
        if mnp_fn_max is not None:
            add_mnp_force_limits(n_m, mnp_fn_max, G, h)
    else:
        print('world can only be vert or planar')
        raise

    # Resize to full matrices.
    G.resize(-1, n_z)
    A.resize(-1, n_z)

    return G.numpy(), h.numpy(), A.numpy(), b.numpy()

