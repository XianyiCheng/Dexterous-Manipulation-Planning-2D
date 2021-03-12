import numpy as np
from .mechanics import *
from pyOpt import Optimization
from pyOpt import SLSQP

def traj_optim_static(paths, tree):
    path, envs, modes, mnps = paths
    guard_index = [0]
    n = len(modes)
    v_init = np.zeros((n,3))
    for i in range(1,n):
        if not np.all(modes[i]==modes[i-1]):
            guard_index.append(i)
        elif len(envs[i])!= 0:
            if not envs[i][0].is_same(envs[i-1][0]):
                guard_index.append(i)
        elif not (mnps[i][0].is_same(mnps[i-1][0]) and mnps[i][1].is_same(mnps[i-1][1])):
            # manipulator change
            guard_index.append(i)
        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(path[i-1])[0:2, 0:2]
        v_init[i-1] = np.dot(g_v.T, np.array(path[i]) - np.array(path[i-1]))
    #guard_index.append(len(modes)-1)
    guard_index = np.unique(guard_index)

    Gs = dict()
    hs = dict()
    As = dict()
    bs = dict()
    for i in range(len(path)):
        G,h,A,b = contact_mode_constraints(path[i], mnps[i], envs[i], modes[i],
                                           tree.world, tree.mnp_mu, tree.env_mu, tree.mnp_fn_max)
        gid = np.any(G[:,0:3],axis=1)
        aid = np.any(A[:,0:3],axis=1)
        Gs[i] = G[gid,0:3]
        hs[i] = h[gid].flatten()
        As[i] = A[aid,0:3]
        bs[i] = b[aid].flatten()

    modeconstraints = (Gs, hs, As, bs)
    q_goal = np.array(tree.x_goal)

    opt_prob = Optimization('Trajectory Optimization', obj_fun)
    x_init = np.hstack((np.array(path).flatten(), v_init.flatten()))
    cs = constraints(x_init, path, Gs, hs, As, bs, guard_index)

    opt_prob.addVarGroup('x', n*6, 'c', value=x_init, lower=-10, upper=10)
    opt_prob.addObj('f')
    opt_prob.addConGroup('g', len(cs), 'i',lower=0.0,upper=10000.0)
    print(opt_prob)
    slsqp = SLSQP()
    #slsqp.setOption('IPRINT', -1)
    slsqp(opt_prob, sens_type='FD', goal=q_goal, path = path, modecons = modeconstraints, guard_index=guard_index)
    print(opt_prob.solution(0))
    qs = [opt_prob.solution(0)._variables[i].value for i in range(n*3)]
    return qs

def obj_fun(x, **kwargs):
    q_goal = kwargs['goal']
    path = kwargs['path']
    Gs, hs, As, bs = kwargs['modecons']
    guard_index = kwargs['guard_index']
    fail = 0
    x = x.flatten()
    f = objective(x,q_goal)
    g = constraints(x,path, Gs, hs, As, bs, guard_index)
    return f, g, fail


def objective(x, q_goal):
    n = int(len(x)/2)
    q = x[0:n]
    v = x[n:]
    f = 0
    for i in range(int(n/3)):
        f += np.dot(v[3*i:3*i+3],v[3*i:3*i+3])
    dq = q[-3:] - q_goal
    f += 10*np.dot(dq,dq)
    return f

def constraints(x, path, Gs, hs, As, bs, guard_index):
    c = []
    n = int(len(x)/2)
    q = x[0:n]
    v = x[n:]
    k = int(n/3)
    for i in range(k):
        q_ = q[3*i:3*i+3]
        v_ = v[3 * i:3 * i + 3]

        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(q_)[0:2, 0:2]
        if i == k-1:
            v_star = np.zeros(3)
        else:
            v_star = np.dot(g_v.T, q[3*i+3:3*i+6] - q_)

        eq_v = (v_ - v_star).flatten()
        ieq_g = (np.dot(Gs[i],v_) - hs[i]).flatten()
        eq_a = (np.dot(As[i],v_) - bs[i]).flatten()
        if i in guard_index:
            eq_guard = (q_ - np.array(path[i])).flatten()
        else:
            eq_guard = np.array([])

        c.append(np.hstack((eq_v,-eq_v,ieq_g,eq_a,-eq_a,eq_guard,-eq_guard)))
    cons = np.hstack(c)
    return cons

