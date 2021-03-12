import random
import numpy as np
import enum
from ..utilities.transformations import *
from ..utilities.geometry import steer
from ..mechanics.mechanics import *
from ..mechanics.stability_margin import *
from .tree import RRTTree, RRTEdge
import time
import scipy.optimize

def qlcp_solver(c,M1,M2,G,h,A,b,nvar,lb,ub):
    b = b.flatten()
    cons = ({'type': 'ineq', 'fun': lambda z: np.dot(M2,z)*np.dot(M1,z)},
            {'type': 'ineq', 'fun': lambda z: np.dot(G,z)+h },
            {'type': 'ineq', 'fun': lambda z: np.dot(M1, z)},
            {'type': 'eq', 'fun': lambda z: np.dot(A,z)+b })
    fun = lambda z: np.dot(z[0:3],z[0:3]) - 2*np.dot(c,z[0:3])
    jac = lambda z: np.concatenate((2*(z[0:3]-c), np.zeros(nvar-3)))
    bnds = scipy.optimize.Bounds(lb=lb,ub=ub)
    res = scipy.optimize.minimize(fun, np.zeros(nvar), method='SLSQP', jac=jac, bounds=bnds, constraints=cons)

    return res

def qlcp(x, v, mnp, env, object_weight, mnp_mu=0.8, env_mu=0.3, mnp_fn_max=None):

    x = np.array(x)
    # Get variable sizes.
    n_m = len(mnp)
    n_e = len(env)
    n_c = n_m + n_e
    n_var = 3 + 4*n_e + 2*n_m

    # Make contact info.
    Ad_gcos, depths, mus = contact_info(mnp, env, mnp_mu, env_mu)
    depths = [0.0]*n_c

    # object gravity
    f_g = np.array([[0.0],[object_weight],[0.0]])
    g_ow = np.linalg.inv(twist_to_transform(x))
    f_o = np.dot(g_ow, f_g)

    n = np.array([[0.0],[1.0],[0.0]])
    D = np.array([[1.0, -1.0],[0.0, 0.0],[0.0, 0.0]])

    # Gx >= h
    G = np.zeros((n_c,n_var))
    h = np.zeros(n_c)
    # Ax = b
    A = np.zeros((3, n_var))
    b = f_o
    M1 = np.zeros((3*n_e,n_var))
    M2 = np.zeros((3*n_e,n_var))
    i_var = 3
    for i in range(n_c):
        N_c = np.dot(Ad_gcos[i].T, n)
        T_c = np.dot(Ad_gcos[i].T, D)
        nAd_gco = np.dot(n.T, Ad_gcos[i]).flatten()
        TAd_gco = np.dot(D.T, Ad_gcos[i])
        if i >= n_e:
            mu = mnp_mu
            A[:, i_var:i_var + 2] = np.hstack((N_c, T_c[:,0].reshape((-1,1))))
            G[i, i_var:i_var + 2] = np.array([mu, -1])
            i_var += 2
        else:
            mu = env_mu
            A[:, i_var:i_var + 3] = np.hstack((N_c, T_c))
            G[i, i_var:i_var + 3] = np.array([mu, -1, -1])
            M1[i*3,0:3] = nAd_gco
            M1[i*3+1:i*3+3, 0:3] = TAd_gco
            M1[i*3+1:i*3+3,6+i*4] = 1
            M2[i*3:(i+1)*3, 3+i*4:3+i*4+4] = 1
            i_var += 4

    lb = np.zeros(n_var)
    lb[0:3] = -np.inf
    3+4*n_e + np.arange(1,2*n_m,2)
    lb[3+4*n_e + np.arange(1,2*n_m,2)] = -np.inf
    ub = np.full(n_var, np.inf)
    if mnp_fn_max is not None:
        ub[3+4*n_e + np.arange(0,2*n_m,2)] = mnp_fn_max

    res = qlcp_solver(v,M1,M2,G,h,A,b,n_var,lb,ub)
    if res.success:
        vx = res.x[0:3]
    else:
        vx = np.zeros(3)
    return vx


class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4

def smallfmod(x, y):
    while x > y:
        x -= y
    while x < 0:
        x += y
    return x

def get_both_velocities(x_rand, x_near):
    x_rand = np.array(x_rand)
    x_rand_ = x_rand[:]
    x = np.array(x_near)
    g_v = np.identity(3)
    g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]

    v_star = np.dot(g_v.T, x_rand - np.array(x))

    if v_star[2] > 0:
        x_rand_[2] = x_rand[2] - 2 * np.pi
    else:
        x_rand_[2] = x_rand[2] + 2 * np.pi

    v_star_ = np.dot(g_v.T, x_rand_ - x)
    return v_star, v_star_

class RRT1(object):
    def __init__(self, X, x_init, x_goal, envir, object, manipulator, max_samples, r=5, world='planar'):
        """
        Template RRTKinodynamic Planner
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.x_init = x_init
        self.x_goal = x_goal
        self.neighbor_radius = r
        self.world = world
        if self.world == 'vert':
            self.object_weight = 10
        else:
            self.object_weight = 0
        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree
        self.environment = envir
        self.object = object
        self.manipulator = manipulator

        self.collision_manager = []
        self.mnp_mu = 0.8
        self.env_mu = 0.3
        self.dist_weight = 1
        self.goal_kch = [1, 1, 1]
        self.cost_weight = [0.2, 1, 1]
        self.step_length = 2
        self.mnp_fn_max = None

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(RRTTree())

    def set_world(self, key):
        self.world = key

    def dist(self, p, q):
        cx = (p[0] - q[0]) ** 2
        cy = (p[1] - q[1]) ** 2
        period = 2 * np.pi
        t1 = smallfmod(p[2], period)
        t2 = smallfmod(q[2], period)
        dt = t2 - t1
        dt = smallfmod(dt + period / 2.0, period) - period / 2.0
        ct = self.dist_weight * dt ** 2
        return cx + cy + ct

    def goal_dist(self, p):
        q = self.x_goal
        cx = (p[0] - q[0]) ** 2
        cy = (p[1] - q[1]) ** 2
        period = 2 * np.pi
        t1 = smallfmod(p[2], period)
        t2 = smallfmod(q[2], period)
        dt = t2 - t1
        dt = smallfmod(dt + period / 2.0, period) - period / 2.0
        ct = dt ** 2
        dist = self.goal_kch[0] * cx + self.goal_kch[1] * cy + self.goal_kch[2] * ct
        return dist

    def get_nearest(self, tree, x):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """
        min_d = np.inf
        for q in self.trees[tree].nodes:
            d = self.dist(q, x)
            if q in self.trees[tree].edges:
                if min_d > d:
                    min_d = d
                    q_near = q
            else:
                self.trees[tree].nodes.remove(q)

        return q_near

    def get_unexpand_nearest(self, tree):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """

        min_d = np.inf
        q_near = self.x_init
        for q in self.trees[tree].nodes:
            if self.trees[tree].goal_expand[q]:
                continue
            d = self.goal_dist(q)
            if q in self.trees[tree].edges:
                if min_d > d:
                    min_d = d
                    q_near = q
            else:
                self.trees[tree].nodes.remove(q)

        return q_near

    def get_goal_nearest(self, tree):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """

        min_d = np.inf
        for q in self.trees[tree].nodes:
            d = self.goal_dist(q)
            if q in self.trees[tree].edges:
                if min_d > d:
                    min_d = d
                    q_near = q
            else:
                self.trees[tree].nodes.remove(q)

        return q_near, min_d

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        n_nodes = 2
        path = [x_goal]
        current = x_goal
        mnp_path = [None]
        key_path = []
        key_mnp_path = []
        if x_init == x_goal:
            return path, mnp_path
        while not self.trees[tree].edges[current].parent == x_init:
            # path.append(self.trees[tree].E[current])
            n_nodes += 1

            key_path.append(self.trees[tree].edges[current].parent)
            key_mnp_path.append(self.trees[tree].edges[current].manip)

            current_path = self.trees[tree].edges[current].path
            path += current_path
            mnp_path += [self.trees[tree].edges[current].manip] * len(current_path)
            current = self.trees[tree].edges[current].parent

            print(current)
        current_path = self.trees[tree].edges[current].path
        path += current_path
        mnp_path += [self.trees[tree].edges[current].manip] * len(current_path)
        key_path.append(self.trees[tree].edges[current].parent)
        key_mnp_path.append(self.trees[tree].edges[current].manip)

        path.append(x_init)
        mnp_path.append(None)

        path.reverse()
        mnp_path.reverse()
        print('number of nodes', n_nodes)
        return path, mnp_path, key_path, key_mnp_path

    def add_waypoints_to_tree(self, tree, edge):
        parent = edge.parent
        path = edge.path[:]
        path.reverse()
        mode = edge.mode
        mnps = edge.manip
        d_i = int(len(path) / 3) + 1
        # print(len(path))
        i = d_i
        while i < len(path):
            x_new = path[i]
            path_i = path[0:i + 1]
            path_i.reverse()
            _, envs = self.check_collision(x_new)
            edge_ = RRTEdge(parent, mnps, envs, path_i, mode)
            self.trees[tree].add(x_new, edge_)
            i += d_i

    def add_collision_manager(self, collision_manager, object, object_shape):
        self.collision_manager = collision_manager
        self.object = object
        self.object_shape = object_shape

    def check_collision(self, x):
        if_collide, w_contacts = self.environment.check_collision(self.object, x)
        contacts = self.object.contacts2objframe(w_contacts, x)

        return if_collide, contacts

    def check_penetration(self, contacts):
        ifPenetrate = False
        for c in contacts:
            if c.d < -0.05:
                ifPenetrate = True
                # print('penetrate')
                break
        return ifPenetrate

    def contact_modes(self, x, envs):
        # TODO: number of manipulator contacts should change according to mnp types
        # _, envs = self.check_collision(x)
        modes = get_contact_modes([Contact([], [], None)] * self.manipulator.npts, envs)
        return modes

    # @profile
    def resample_manipulator_contacts(self, tree, x):
        # mnp = self.object.sample_contacts(1)
        # ifReturn = True
        # mnp_config = None

        pre_mnp = self.trees[tree].edges[x].manip
        num_manip = self.manipulator.npts
        ifReturn = False
        mnp_config = None
        if pre_mnp is None:
            while not ifReturn:
                mnp = self.object.sample_contacts(num_manip)
                isReachable, mnp_config = self.manipulator.inverse_kinematics(mnp)
                # ifCollide, _ = self.environment.check_collision(self.manipulator, mnp_config)
                ifCollide = self.manipulator.if_collide_w_env(self.environment, mnp_config, x)
                ifReturn = isReachable and (not ifCollide)
            return ifReturn, mnp, mnp_config

        else:
            counter = 0
            max_count = 4
            # ifReturn = False
            while counter < max_count:
                counter += 1
                mnp = np.array([None] * num_manip)
                # random find contacts that change
                num_manip_left = random.randint(0, num_manip - 1)
                manip_left = random.sample(range(num_manip), num_manip_left)

                # check if equilibrium if the selected manip contacts are moved
                if static_equilibrium(x, np.array(pre_mnp)[manip_left], self.trees[tree].edges[x].env, self.world,
                                      self.mnp_mu, self.env_mu, self.mnp_fn_max):
                    # randomly sample manipulator contacts
                    mnp[manip_left] = np.array(pre_mnp)[manip_left]
                    for i in range(len(mnp)):
                        if mnp[i] is None:
                            mnp[i] = self.object.sample_contacts(1)[0]
                    # check inverse kinematics
                    isReachable, mnp_config = self.manipulator.inverse_kinematics(mnp)
                    if isReachable:
                        ifCollide = self.manipulator.if_collide_w_env(self.environment, mnp_config, x)
                        # if (mnp_config[3] < 0 or mnp_config[1] < 0) and not ifCollide:
                        #     print('sampled!',x)
                        if not ifCollide:
                            ifReturn = True
                            break
        if ifReturn and mnp[0] is None:
            print('sth wrong with resample manipulator contacts')
        return ifReturn, mnp, mnp_config

    # @profile
    def best_mnp_location(self, tree, x_near, x_rand, vel):

        n_sample = 5

        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(x_near)[0:2, 0:2]
        v_star = np.dot(g_v.T, np.array(x_rand) - np.array(x_near))

        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip
        mnps_list = []
        score_list = []
        dist_list = []
        if mnps is not None:
            mnps_list.append(mnps)

            score = 0
            score_list.append(score)
            v = self.inverse_mechanics(x_near, v_star, envs, mnps)
            dist_list.append(self.dist(v, v_star))

        for i in range(n_sample):
            ifsampled, mnps, _ = self.resample_manipulator_contacts(tree, x_near)
            if ifsampled:
                v = self.inverse_mechanics(x_near, v_star, envs, mnps)
                if np.linalg.norm(v) > 1e-3:
                    mnps_list.append(mnps)
                    score = 0.1
                    score_list.append(score)
                    dist_list.append(self.dist(v, v_star))
                    break

        # how to choose the best manipulator location, what's the metric?

        if len(mnps_list) > 0:
            best_score_ind = np.argmax(score_list)
            # print('manipulator changed')
            return mnps_list[best_score_ind]
        else:
            return None

    def inverse_mechanics(self, x, v_star, envs, mnps):

        v = qlcp(x, v_star, mnps, envs, self.object_weight,self.mnp_mu,self.env_mu,self.mnp_fn_max)
        # v = qp_inv_mechanics_2d(np.array(v_star), np.array(x), mnps, envs, mode, self.world, self.mnp_mu, self.env_mu,
        #                         self.mnp_fn_max)
        return v

    def forward_integration(self, x_near, x_rand, envs, mnps):
        # all collision checking, event detection , timestepping, ...
        counter = 0
        h = 0.2
        Status_manipulator_collide = False

        x_rand = np.array(x_rand)
        x = np.array(x_near)
        path = [tuple(x)]  # TODO: hack
        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]

        v_star = np.dot(g_v.T, x_rand - np.array(x))

        v = self.inverse_mechanics(x, v_star, envs, mnps)
        if np.linalg.norm(v) < 1e-3:
            if v_star[2] > 0:
                x_rand[2] = x_rand[2] - 2 * np.pi
            else:
                x_rand[2] = x_rand[2] + 2 * np.pi
            v_star = np.dot(g_v.T, x_rand - np.array(x))
            v = self.inverse_mechanics(x, v_star, envs, mnps)
            if np.linalg.norm(v) < 1e-3:
                return tuple(x), path, Status_manipulator_collide

        max_counter = int(np.linalg.norm(v_star) / h) * 10
        if np.linalg.norm(v_star) > h:
            v_star = steer(0, v_star, h)

        # TODO: velocity-mode-projection: v_star = v_star*d_proj

        while np.linalg.norm(v_star) > 1e-2 and counter < max_counter:
            g_v = np.identity(3)
            g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]
            counter += 1
            # v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
            v = self.inverse_mechanics(x, v_star, envs, mnps)
            if np.linalg.norm(v) < 1e-3:
                break

            # check collision
            x_ = x.flatten() + np.dot(g_v, v).flatten()
            _, mnp_config = self.manipulator.inverse_kinematics(mnps)  # TODO: need to store mnp_config, no IK everytime
            if self.manipulator.if_collide_w_env(self.environment, mnp_config, x_):
                Status_manipulator_collide = True
                break
            if_collide, contacts = self.check_collision(x_)
            ifpenetrate = self.check_penetration(contacts)
            # ifpenetrate = False
            if not ifpenetrate:
                # update x if not collide
                if not static_equilibrium(x, mnps, contacts, self.world,
                                      self.mnp_mu, self.env_mu, self.mnp_fn_max):
                    break
                x = x_
                path.append(tuple(x))
                envs = contacts
                g_v = np.identity(3)
                g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]
                v_star = np.dot(g_v.T, x_rand - x)
                if np.linalg.norm(v_star) > h:
                    v_star = steer(0, v_star, h)

            else:
                # return x backward at collisiton
                depths = np.array([c.d for c in contacts])
                max_i = np.argmax(depths)
                d_max = np.max(-depths)
                p_max = contacts[max_i].p
                n_max = contacts[max_i].n
                # vs = np.dot(adjointTrans_2d(config2trans(x)), v)
                # v_p_max = np.dot(v_hat(vs), np.concatenate([p_max, [1]]))
                v_p_max = np.dot(v_hat(v), np.concatenate([p_max, [1]]))
                k_new = 1 - d_max / abs(np.dot(v_p_max[0:2], n_max))
                if abs(k_new) < 1:
                    v = k_new * v
                    x = x + np.dot(g_v, v).flatten()
                    path.append(tuple(x))
                break
        path_ = [path[0]]
        for i in range(len(path)):
            if np.linalg.norm(np.array(path_[-1]) - np.array(path[i])) > h / 5:
                path_.append(path[i])

        if path[-1] not in path_:
            path_.append(path[-1])

        if np.linalg.norm(x) > 1000:
            print('something wrong')

        return tuple(x), path_, Status_manipulator_collide

    def is_feasible_velocity(self, tree, x_near, x_rand, mode):
        # check is there posive feasible velocity to the x-rand under this mode
        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip

        x_rand = np.array(x_rand)
        x_rand_ = x_rand[:]
        x = np.array(x_near)
        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]

        v_star = np.dot(g_v.T, x_rand - np.array(x))

        if v_star[2] > 0:
            x_rand_[2] = x_rand[2] - 2 * np.pi
        else:
            x_rand_[2] = x_rand[2] + 2 * np.pi

        v_star_ = np.dot(g_v.T, x_rand_ - x)

        if len(envs) == 0:
            return True, v_star, v_star_
        if mnps is None:
            mnps = []
            mode = mode[(len(mode) - len(envs)):]

        d_proj = velocity_project_direction(v_star, mnps, envs, mode)
        d_proj_ = velocity_project_direction(v_star_, mnps, envs, mode)

        is_feasible = (np.linalg.norm(d_proj) > 1e-3) or (np.linalg.norm(d_proj_) > 1e-3)

        return is_feasible, d_proj, d_proj_

    # @profile
    def extend(self, tree, x_near, x_rand, vel):

        # h = self.step_length

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip

        # manipulator contacts: sample from the previous one or new mnps
        # Todo: change manipulator location when stability margin is low
        if (self.trees[tree].edges[x_near].manip is None) \
                or self.trees[tree].edges[x_near].manipulator_collide:
            # Todo: sample good manipulator location given conact mode
            mnps = self.best_mnp_location(tree, x_near, x_rand, vel)

        if mnps is None:
            return x_near, Status.TRAPPED, None


        # forward (ITM)
        x_new, path, status_mnp_collide = self.forward_integration(x_near, x_rand, envs, mnps)
        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED

        elif self.dist(x_rand, x_new) < 2e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        edge = None
        if status != Status.TRAPPED:
            path.reverse()
            _, new_envs = self.check_collision(x_new)
            edge = RRTEdge(x_near, mnps, new_envs, path, None)
            edge.manipulator_collide = status_mnp_collide

        return x_new, status, edge

    # @profile
    def extend_changemnp(self, tree, x_near, x_rand, vel):

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        # manipulator contacts: sample from the previous one or new mnps
        mnps = self.best_mnp_location(tree, x_near, x_rand,  vel)

        if mnps is None:
            return x_near, Status.TRAPPED, None

        # forward (ITM)
        x_new, path, status_mnp_collide = self.forward_integration(x_near, x_rand, envs, mnps)
        path.reverse()
        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED
        elif self.dist(x_rand, x_new) < 2e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        edge = None
        if status != Status.TRAPPED:
            _, envs = self.check_collision(x_new)
            edge = RRTEdge(x_near, mnps, envs, path, None)
            edge.manipulator_collide = status_mnp_collide


        return x_new, status, edge

    # @profile
    def search(self, init_mnp=None):
        t_start = time.time()
        _, init_envs = self.check_collision(self.x_init)
        edge = RRTEdge(None, init_mnp, init_envs, None, None)
        self.trees[0].add(self.x_init, edge)
        # init_modes = self.contact_modes(self.x_init, init_envs)
        # self.trees[0].add_mode_enum(self.x_init, init_modes)
        start_flag = True
        while self.samples_taken < self.max_samples:
            x_nearest, d_nearest = self.get_goal_nearest(0)
            if d_nearest < 0.04:
                print('GOAL REACHED. Nearest state: ', x_nearest, ', dist: ', d_nearest)
                paths = self.reconstruct_path(0, self.x_init, x_nearest)
                return paths, True, self.samples_taken

            if random.randint(0, 3) == 1 and not start_flag:
                # x_rand = self.x_goal
                x_rand = self.X.sample_free()
                x_near = self.get_nearest(0, x_rand)
            else:
                start_flag = False
                x_rand = self.x_goal
                x_near = self.get_unexpand_nearest(0)
                self.trees[0].goal_expand[x_near] = True

            near_envs = self.trees[0].edges[x_near].env
            # if x_near in self.trees[0].enum_modes:
            #     near_modes = self.trees[0].enum_modes[x_near]
            # else:
            #     near_modes = self.contact_modes(x_near, near_envs)
            #     self.trees[0].add_mode_enum(x_near, near_modes)

            v_star, v_star_ = get_both_velocities(x_rand, x_near)
            if self.trees[0].edges[x_near].manip is None:
                ifextend = True
                v = v_star
            else:
                v_pre = self.inverse_mechanics(x_near, v_star, near_envs, self.trees[0].edges[x_near].manip)
                v_pre_ = self.inverse_mechanics(x_near, v_star_, near_envs, self.trees[0].edges[x_near].manip)
                if np.linalg.norm(v_pre) > 1e-3:
                    v = v_pre
                    ifextend = True
                elif np.linalg.norm(v_pre_) > 1e-3:
                    v = v_pre
                    ifextend = True
                else:
                    v = v_star
                    ifextend = False

            if ifextend:
                x_new, status, edge = self.extend(0, x_near, x_rand, v)
            else:
                x_new, status, edge = self.extend_changemnp(0, x_near, x_rand, v)
            if status != Status.TRAPPED and self.dist(x_new, self.get_nearest(0, x_new)) > 1e-3:
                self.trees[0].add(x_new, edge)
                self.samples_taken += 1
                print('sample ', self.samples_taken, ', x: ', x_new)
                self.add_waypoints_to_tree(0, edge)

            t_end = time.time()
            if t_end - t_start > self.max_time:
                break
        x_nearest, d_nearest = self.get_goal_nearest(0)
        print('GOAL NOT REACHED. Nearest state: ', x_nearest, ', dist: ', d_nearest)
        paths = self.reconstruct_path(0, self.x_init, x_nearest)

        return paths, False, self.samples_taken




