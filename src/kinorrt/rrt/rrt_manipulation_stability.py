import random
import numpy as np
import enum
from ..utilities.transformations import *
from ..utilities.geometry import steer
from ..mechanics.mechanics import *
from ..mechanics.stability_margin import *
from .tree import RRTTree, RRTEdge
import time
from itertools import product


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

def enumerate_hand_modes(n_mnp):
    if n_mnp == 1:
        h_modes = np.array([CONTACT_MODE.LIFT_OFF, CONTACT_MODE.STICKING,
                            CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.SLIDING_LEFT]).reshape(-1,1)
    elif n_mnp == 2:
        h_modes = np.array([[CONTACT_MODE.STICKING,CONTACT_MODE.STICKING],
                           [CONTACT_MODE.SLIDING_RIGHT,CONTACT_MODE.SLIDING_RIGHT],
                            [CONTACT_MODE.SLIDING_LEFT,CONTACT_MODE.SLIDING_LEFT],
                            [CONTACT_MODE.STICKING,CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.STICKING],
                            [CONTACT_MODE.SLIDING_LEFT,CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.SLIDING_RIGHT,CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.LIFT_OFF,CONTACT_MODE.SLIDING_RIGHT],
                            [CONTACT_MODE.LIFT_OFF,CONTACT_MODE.SLIDING_LEFT]])
    return h_modes

class RRTManipulationStability(object):
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
        self.h_modes = enumerate_hand_modes(self.manipulator.npts)

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(RRTTree())

    def set_world(self, key):
        self.world = key

    def initialize_stability_margin_solver(self, solver):
        self.smsolver = solver

    def dist(self, p, q):
        cx = self.dist_weight[0]*(p[0] - q[0]) ** 2
        cy = self.dist_weight[1]*(p[1] - q[1]) ** 2
        period = 2 * np.pi
        t1 = smallfmod(p[2], period)
        t2 = smallfmod(q[2], period)
        dt = t2 - t1
        dt = smallfmod(dt + period / 2.0, period) - period / 2.0
        ct = self.dist_weight[2] * dt ** 2
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
        path = []
        current = x_goal
        print(current, self.trees[tree].edges[current].mode, self.trees[tree].edges[current].score)
        mnp_path = []
        velocity_path = []
        mode_path = []
        env_path = []

        if x_init == x_goal:
            return path, mnp_path

        while not self.trees[tree].edges[current].parent == x_init:
            # path.append(self.trees[tree].E[current])
            n_nodes += 1

            current_path = self.trees[tree].edges[current].path
            path += current_path
            mnp_path += [self.trees[tree].edges[current].manip] * len(current_path)
            pp = self.trees[tree].edges[current].paths[1]
            pp.reverse()
            velocity_path += pp
            pp = self.trees[tree].edges[current].paths[2]
            pp.reverse()
            env_path += pp
            pp = self.trees[tree].edges[current].paths[3]
            pp.reverse()
            mode_path += pp
            current = self.trees[tree].edges[current].parent

            print(current, self.trees[tree].edges[current].mode, self.trees[tree].edges[current].score)
        current_path = self.trees[tree].edges[current].path
        path += current_path
        mnp_path += [self.trees[tree].edges[current].manip] * len(current_path)
        pp = self.trees[tree].edges[current].paths[1]
        pp.reverse()
        velocity_path += pp
        pp = self.trees[tree].edges[current].paths[2]
        pp.reverse()
        env_path += pp
        pp = self.trees[tree].edges[current].paths[3]
        pp.reverse()
        mode_path += pp
        print(current, self.trees[tree].edges[current].mode, self.trees[tree].edges[current].score)
        #path.append(x_init)
        #mnp_path.append(None)

        path.reverse()
        mnp_path.reverse()
        velocity_path.reverse()
        env_path.reverse()
        mode_path.reverse()

        print('number of nodes', n_nodes)
        return path, velocity_path, mnp_path, env_path, mode_path

    def add_waypoints_to_tree(self, tree, edge):
        parent = edge.parent
        path = edge.path[:]
        paths = edge.paths
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
            paths_i = (paths[0][0:i+1],paths[1][0:i+1],paths[2][0:i+1],paths[3][0:i+1])
            _, envs = self.check_collision(x_new)
            edge_ = RRTEdge(parent, mnps, envs, path_i, mode)
            edge_.score = edge.score
            edge_.paths = paths_i
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
    def best_mnp_location(self, tree, x_near, x_rand, mode, vel):

        n_sample = 1

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
            # score = self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode,
            #                                                self.object_weight, self.mnp_fn_max, self.dist_weight)
            score = 0
            score_list.append(score)
            v = self.inverse_mechanics(x_near, v_star, envs, mnps, mode)
            dist_list.append(self.dist(v, v_star))

        for i in range(n_sample):
            ifsampled, mnps, _ = self.resample_manipulator_contacts(tree, x_near)
            # check if key points sampled
            # ep0 = np.array([envs[0].p[0], envs[1].p[0]])
            # mm = np.array(mode[2:])
            # if ifsampled and x_near[0]<0 and (mnps[0].p[1] < 0 or mnps[1].p[1] < 0) \
            #         and mm[ep0>0] == CONTACT_MODE.STICKING and mm[ep0<0] == CONTACT_MODE.LIFT_OFF:
            #     print('sampled')
            if ifsampled:
                v = self.inverse_mechanics(x_near, v_star, envs, mnps, mode)
                if np.linalg.norm(v) > 1e-3:
                    mnps_list.append(mnps)

                    # score = self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode,
                    #                                               10, self.mnp_fn_max, self.dist_weight)
                    score = 0.1
                    score_list.append(score)
                    dist_list.append(self.dist(v, v_star))

        # how to choose the best manipulator location, what's the metric?

        if len(mnps_list) > 0:
            best_score_ind = np.argmax(score_list)
            # print('manipulator changed')
            return mnps_list[best_score_ind]
        else:
            return None

    def inverse_mechanics(self, x, v_star, envs, mnps, mode):
        if mode is None:
            print('mode cannot be None for RRTKino_w_modes class')
            raise

        # mnps = [(np.array(m.p), np.array(m.n), m.d) for m in mnps]
        # envs = [(np.array(m.p), np.array(m.n), m.d) for m in envs]
        v = qp_inv_mechanics_2d(np.array(v_star), np.array(x), mnps, envs, mode, self.world, self.mnp_mu, self.env_mu,
                                self.mnp_fn_max)
        return v

    def forward_integration(self, x_near, x_rand, envs, mnps, mode):
        # all collision checking, event detection , timestepping, ...
        counter = 0
        h = 0.2
        Status_manipulator_collide = False

        x_rand = np.array(x_rand)
        x = np.array(x_near)
        path = [tuple(x)]  # TODO: hack
        env_path = [envs]
        mode_path = [mode]
        velocity_path = []
        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]

        v_star = np.dot(g_v.T, x_rand - np.array(x))

        v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
        if np.linalg.norm(v) < 1e-3:
            if v_star[2] > 0:
                x_rand[2] = x_rand[2] - 2 * np.pi
            else:
                x_rand[2] = x_rand[2] + 2 * np.pi
            v_star = np.dot(g_v.T, x_rand - np.array(x))
            v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
            if np.linalg.norm(v) < 1e-3:
                return tuple(x), path, Status_manipulator_collide,[]

        max_counter = int(np.linalg.norm(v_star) / h) * 10
        if np.linalg.norm(v_star) > h:
            v_star = steer(0, v_star, h)

        # TODO: velocity-mode-projection: v_star = v_star*d_proj

        d_proj = velocity_project_direction(v_star, mnps, envs, mode)
        if sum(d_proj) == 0:
            d_proj = v_star / np.linalg.norm(v_star)
        v_star_proj = np.dot(v_star, d_proj) * d_proj

        #v_star_proj = v_star
        # finger mode
        finger_mode = np.array(mode[0:len(mnps)])
        if CONTACT_MODE.LIFT_OFF in finger_mode:
            cm = len(mnps)
            remain_idx = finger_mode != CONTACT_MODE.LIFT_OFF
            mnps = list(np.array(mnps)[remain_idx])
            finger_mode = finger_mode[remain_idx]
            mode = list(finger_mode) + mode[cm:]

        while np.linalg.norm(v_star) > 1e-2 and counter < max_counter:
            g_v = np.identity(3)
            g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]
            counter += 1
            # v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
            v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
            if np.linalg.norm(v) < 1e-3:
                v = self.inverse_mechanics(x, v_star_proj, envs, mnps, mode)
                if np.linalg.norm(v) < 1e-3:
                    break

            # finger mode
            for i_finger in range(len(mnps)):
                fm = finger_mode[i_finger]
                if fm == CONTACT_MODE.SLIDING_LEFT:
                    mnps[i_finger].p = []
                    pass
                elif fm == CONTACT_MODE.SLIDING_RIGHT:
                    mnps[i_finger].p = []
                    pass

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

                # check the number of envs contacts
                if len(envs) != len(contacts):
                    if len(contacts) == (sum(np.array(mode) != CONTACT_MODE.LIFT_OFF) - len(mnps)):
                        mode = list(np.array(mode)[np.array(mode) != CONTACT_MODE.LIFT_OFF])
                    else:
                        x = x_
                        path.append(tuple(x))
                        velocity_path.append(v)
                        env_path.append(contacts)
                        mode_path.append(mode)
                        break
                else:
                    is_same_contacts = True
                    for i in range(len(envs)):
                        if not envs[i].is_same(contacts[i]):
                            is_same_contacts = False
                    if not is_same_contacts:
                        break

                x = x_
                path.append(tuple(x))
                velocity_path.append(v)
                env_path.append(contacts)
                mode_path.append(mode)
                envs = contacts
                g_v = np.identity(3)
                g_v[0:2, 0:2] = config2trans(x)[0:2, 0:2]
                v_star = np.dot(g_v.T, x_rand - x)
                if np.linalg.norm(v_star) > h:
                    v_star = steer(0, v_star, h)
                v_star_proj = np.dot(v_star, d_proj) * d_proj
                #v_star_proj = v_star
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
                    _, contacts = self.check_collision(x)
                    path.append(tuple(x))
                    velocity_path.append(v)
                    env_path.append(contacts)
                    mode_path.append(mode)
                elif d_max < 0.1:
                    x = x + np.dot(g_v, v).flatten()
                    path.append(tuple(x))
                    velocity_path.append(v)
                    env_path.append(contacts)
                    mode_path.append(mode)
                break
        velocity_path.append(np.zeros(3))
        path_ = [path[0]]
        velocity_path_ = [velocity_path[0]]
        env_path_ = [env_path[0]]
        mode_path_ = [mode_path[0]]
        for i in range(len(path)):
            if np.linalg.norm(np.array(path_[-1]) - np.array(path[i])) > 0.1:
                path_.append(path[i])
                velocity_path_ .append(velocity_path[i])
                env_path_.append(env_path[i])
                mode_path_.append(mode_path[i])

        if path[-1] not in path_:
            path_.append(path[-1])
            velocity_path_.append(velocity_path[i])
            env_path_.append(env_path[i])
            mode_path_.append(mode_path[i])

        if np.linalg.norm(x) > 1000:
            print('something wrong')
        paths = (path_, velocity_path_, env_path_, mode_path_)

        return tuple(x), path_, Status_manipulator_collide, paths

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
    def extend_w_mode(self, tree, x_near, x_rand, mode, v, v_):
        # Todo: use stability margin to choose ?

        if np.linalg.norm(v) > 1e-4:
            vel = v

        else:
            vel = v_
        # h = self.step_length

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip

        # manipulator contacts: sample from the previous one or new mnps
        # Todo: change manipulator location when stability margin is low
        if (self.trees[tree].edges[x_near].manip is None) \
                or self.trees[tree].edges[x_near].manipulator_collide:
            # Todo: sample good manipulator location given conact mode
            mnps = self.best_mnp_location(tree, x_near, x_rand, mode, vel)

        if mnps is None:
            return x_near, Status.TRAPPED, None, 0.0


        # forward (ITM)
        x_new, path, status_mnp_collide, paths = self.forward_integration(x_near, x_rand, envs, mnps, mode)
        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED

        elif self.dist(x_rand, x_new) < 2e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        stability_margin_score = 0.0
        edge = None
        if status != Status.TRAPPED:
            if len(envs)>0:
                e_modes = np.array(get_contact_modes([], self.trees[tree].edges[x_near].env))
                e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

                stability_margin_score = self.smsolver.compute_stability_margin(x_near, vel, self.env_mu, self.mnp_mu, self.trees[tree].edges[x_near].env, mnps, mode, e_modes, self.h_modes,
                                                      self.object_weight, self.mnp_fn_max)[0]


            path.reverse()
            _, new_envs = self.check_collision(x_new)
            edge = RRTEdge(x_near, mnps, new_envs, path, mode)
            edge.manipulator_collide = status_mnp_collide
            edge.score = stability_margin_score
            edge.paths = paths
            if stability_margin_score < 0:
                print('zero score')
        return x_new, status, edge, stability_margin_score

    # @profile
    def extend_w_mode_changemnp(self, tree, x_near, x_rand, mode, v, v_):
        # Todo: use stability margin to choose ?

        # h = self.step_length
        # if np.linalg.norm(np.array(x_near) - np.array(x_rand)) > h:
        #     x_rand = steer(x_near, x_rand, h)

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        # manipulator contacts: sample from the previous one or new mnps
        # Todo: change manipulator location when stability margin is low
        if np.linalg.norm(v) > 1e-4:
            mnps = self.best_mnp_location(tree, x_near, x_rand, mode, v)
            vel = v
        else:
            mnps = self.best_mnp_location(tree, x_near, x_rand, mode, v_)
            vel = v_

        if mnps is None:
            return x_near, Status.TRAPPED, None, 0.0


        stability_margin_score = 0.0

        # forward (ITM)
        x_new, path, status_mnp_collide, paths = self.forward_integration(x_near, x_rand, envs, mnps, mode)
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

            if len(envs) > 0:

                e_modes = np.array(get_contact_modes([], self.trees[tree].edges[x_near].env))
                e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

                stability_margin_score = self.smsolver.compute_stability_margin(x_near, vel, self.env_mu, self.mnp_mu, self.trees[tree].edges[x_near].env, mnps, mode, e_modes, self.h_modes,
                                                      self.object_weight, self.mnp_fn_max)[0]

            edge = RRTEdge(x_near, mnps, envs, path, mode)
            edge.manipulator_collide = status_mnp_collide
            edge.score = stability_margin_score
            edge.paths = paths

        return x_new, status, edge, stability_margin_score

    # @profile
    def search(self, init_mnp=None):
        _, init_envs = self.check_collision(self.x_init)
        edge = RRTEdge(None, init_mnp, init_envs, None, None)
        self.trees[0].add(self.x_init, edge)
        init_modes = self.contact_modes(self.x_init, init_envs)
        self.trees[0].add_mode_enum(self.x_init, init_modes)
        start_flag = True
        while self.samples_taken < self.max_samples:
            x_nearest, d_nearest = self.get_goal_nearest(0)
            if d_nearest < 0.04:
                print('GOAL REACHED. Nearest state: ', x_nearest, ', dist: ', d_nearest)
                paths = self.reconstruct_path(0, self.x_init, x_nearest)
                return paths

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
            if len(near_envs) >= 3:
                print('debug here')
            if x_near in self.trees[0].enum_modes:
                near_modes = self.trees[0].enum_modes[x_near]
            else:
                near_modes = self.contact_modes(x_near, near_envs)
                self.trees[0].add_mode_enum(x_near, near_modes)

            for m in near_modes:
                is_feasible, v, v_ = self.is_feasible_velocity(0, x_near, x_rand, m)
                if not is_feasible:
                    # print('velocity infeasible for this mode')
                    continue
                elif self.trees[0].edges[x_near].manip is not None:
                    v_pre = self.inverse_mechanics(x_near, v, near_envs, self.trees[0].edges[x_near].manip, m)
                    v_pre_ = self.inverse_mechanics(x_near, v_, near_envs, self.trees[0].edges[x_near].manip, m)
                    ifextend = np.linalg.norm(v_pre) > 1e-3 or np.linalg.norm(v_pre_) > 1e-3
                else:
                    ifextend = True

                if ifextend:
                    x_new, status, edge, score = self.extend_w_mode(0, x_near, x_rand, m, v, v_)
                else:
                    x_new, status, edge, score = self.extend_w_mode_changemnp(0, x_near, x_rand, m, v, v_)
                if status != Status.TRAPPED and self.dist(x_new, self.get_nearest(0, x_new)) > 1e-3:# and score >=0:
                    self.trees[0].add(x_new, edge)
                    self.samples_taken += 1
                    print('sample ', self.samples_taken, ', x: ', x_new, 'mode', m, 'score', score)
                    self.add_waypoints_to_tree(0, edge)

        x_nearest, d_nearest = self.get_goal_nearest(0)
        print('GOAL NOT REACHED. Nearest state: ', x_nearest, ', dist: ', d_nearest)
        paths = self.reconstruct_path(0, self.x_init, x_nearest)

        return paths




