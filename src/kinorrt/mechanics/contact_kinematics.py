import numpy as np
from .mechanics import *
from ..utilities.transformations import *
import itbl._itbl as _itbl

def RectangleBox(width, length):
    return _itbl.Rectangle(width, length, 2, 0.0)

class point_manipulator(object):
    def __init__(self):
        self.obj = RectangleBox(0.15,0.15)
        self.npts = 1

    def contacts2mnpframe(self, w_contacts, x):
        return [(0,0)]

    def forward_kinematics(self, config):
        return [config]

    def inverse_kinematics(self, contacts):
        # contacts: contacts w.r.t the object
        isReachable = True
        config = contacts[0].p
        return isReachable, config

    def update_config(self, config, part_config):
        p0 = config[0:2]
        p0 = np.dot(config2trans(np.array(part_config)), np.array(list(p0) + [1]))
        p0 = p0[0:2]
        T30 = np.identity(4)
        T30[0, 3] = p0[0]
        T30[1, 3] = p0[1]
        self.obj.transform()[:, :] = T30
        return

    def if_collide_w_env(self, envir, config, part_config):
        if_collide = False
        self.update_config(config, part_config)
        manifold0 = envir.collision_manager.collide(self.obj)

        if len(manifold0.depths) != 0:
            d = min(manifold0.depths)
            if d < -1e-3:
                if_collide = True
        return if_collide

    def score_collide_w_env(self, envir, config, part_config):
        self.update_config(config, part_config)
        manifold0 = envir.collision_manager.collide(self.obj)
        if len(manifold0.depths)  == 0:
            score = 0.1
        else:
            score = np.min(manifold0.depths)
        return score

class doublepoint_manipulator(object):
    def __init__(self, config_bounds = None):
        self.obj = [RectangleBox(0.2,0.2), RectangleBox(0.2,0.2)]
        self.npts = 2
        self.config_bounds = config_bounds
    def contacts2mnpframe(self, w_contacts, x):
        return None

    def forward_kinematics(self, config):
        return [config]

    def inverse_kinematics(self, contacts):
        # contacts: contacts w.r.t the object
        isReachable = True
        config = list(contacts[0].p) + list(contacts[1].p)
        return isReachable, config

    def update_config(self, config, part_config):
        p0 = config[0:2]
        p0 = np.dot(config2trans(np.array(part_config)), np.array(list(p0) + [1]))
        p0 = p0[0:2]
        T30 = np.identity(4)
        T30[0,3] = p0[0]
        T30[1,3] = p0[1]
        self.obj[0].transform()[:, :] = T30

        p1 = config[2:]
        p1 = np.dot(config2trans(np.array(part_config)), np.array(list(p1) + [1]))
        p1 = p1[0:2]
        T31 = np.identity(4)
        T31[0,3] = p1[0]
        T31[1,3] = p1[1]
        self.obj[1].transform()[:, :] = T31
        return

    def if_collide_w_env(self, envir, config, part_config):
        if_collide = False
        self.update_config(config, part_config)
        manifold0 = envir.collision_manager.collide(self.obj[0])
        manifold1 = envir.collision_manager.collide(self.obj[1])
        if len(manifold0.depths) != 0:
            if_collide = min(manifold0.depths) < -1e-3
        if not if_collide and len(manifold1.depths) != 0:
            if_collide = min(manifold1.depths) < -1e-3
        if self.config_bounds is not None:
            p1 = self.obj[0].transform()[0:2,-1]
            p2 = self.obj[1].transform()[0:2,-1]
            p = np.array(list(p1) + list(p2))
            if np.any(p < self.config_bounds[0]) or np.any(p > self.config_bounds[1]):
                if_collide = True

        return if_collide

    def score_collide_w_env(self, envir, config, part_config):
        self.update_config(config, part_config)
        manifold0 = envir.collision_manager.collide(self.obj[0])
        manifold1 = envir.collision_manager.collide(self.obj[1])
        if (len(manifold0.depths) + len(manifold1.depths)) == 0:
            score = 0.1
        else:
            d = list(manifold0.depths) + list(manifold1.depths)
            score = np.min(d)
        return score

class part(object):
    def __init__(self, obj, object_shape, allow_contact_edges = [True]*4):
        self.obj = obj
        self.object_shape = object_shape
        sides = np.array([0,1,2,3])
        self.sides = sides[allow_contact_edges]

    def update_config(self, x):
        T2 = config2trans(np.array(x))
        T3 = np.identity(4)
        T3[0:2, 3] = T2[0:2, 2]
        T3[0:2, 0:2] = T2[0:2, 0:2]
        self.obj.transform()[:, :] = T3
        return

    def contacts2objframe(self, w_contacts, x):
        contacts = []
        g_inv = inv_g_2d(config2trans(np.array(x)))
        # the contacts are wrt the object frame
        for c in w_contacts:
            cp = np.array(c.p)
            cn = np.array(c.n)
            cp_o = np.dot(g_inv,np.concatenate([cp,[1]]))
            cn_o = np.dot(g_inv[0:2,0:2], cn)
            ci = Contact(cp_o[0:2], cn_o, c.d)
            contacts.append(ci)
        return contacts

    def sample_contacts(self, npts):
        return sample_finger_contact_box(self.object_shape, self.sides, npts)

class environment(object):
    def __init__(self, collision_manager):
        self.collision_manager = collision_manager

    def check_collision(self, target, target_config):
        # contacts are in the world frame
        target.update_config(target_config)
        manifold = self.collision_manager.collide(target.obj)
        if_collide = sum(np.array(manifold.depths) < 0.015) != 0

        n_pts = len(manifold.contact_points)
        contacts = []

        # the contacts are wrt the object frame
        for i in range(n_pts):
            if manifold.depths[i] >= 0.015:
                continue
            cp = manifold.contact_points[i]
            cn = manifold.normals[i]
            ci = Contact(cp, cn, manifold.depths[i])
            contacts.append(ci)

        return if_collide, contacts


def sample_finger_contact_box(shape, sides = [0,1,2,3], npts = 1):
    Half_L_outer = shape[0]
    Half_W_outer = shape[1]
    Half_L_inner = shape[2]
    Half_W_inner = shape[3]

    contacts = []

    finger_sides = np.random.choice(sides, npts)
    for side in finger_sides:
        if side == 0:
            n = np.array([0,-1])
            p = np.array([2*Half_L_outer*(np.random.random() - 0.5), Half_W_outer])
        elif side == 1:
            n = np.array([-1,0])
            p = np.array([Half_L_outer, 2*Half_W_outer*(np.random.random() - 0.5)])
        elif side == 2:
            n = np.array([0,1])
            p = np.array([2*Half_L_outer*(np.random.random() - 0.5),-Half_W_outer])
        elif side == 3:
            n = np.array([1,0])
            p = np.array([-Half_L_outer, 2*Half_W_outer*(np.random.random() - 0.5)])
        elif side == 4:
            n = np.array([0, 1])
            p = np.array([ 2 * Half_L_inner * (np.random.random() - 0.5),Half_W_inner])
        elif side == 5:
            n = np.array([1, 0])
            p = np.array([Half_L_inner,2 * Half_W_inner * (np.random.random() - 0.5)])
        elif side == 6:
            n = np.array([0, -1])
            p = np.array([ 2 * Half_L_inner * (np.random.random() - 0.5), -Half_W_inner])
        elif side == 7:
            n = np.array([-1, 0])
            p = np.array([-Half_L_inner, 2 * Half_W_inner * (np.random.random() - 0.5)])
        contact = Contact(p,n,0.0)
        contacts.append(contact)
    return contacts