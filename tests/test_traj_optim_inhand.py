
from time import time

import glm
from itbl import Ray, Shader
from itbl.accelerators import SDF, BVHAccel
from itbl.cameras import TrackballCamera

from itbl.shapes import Box
from itbl.util import get_color, get_data
from itbl.viewer import Application, Viewer
from itbl.viewer.backend import *
import itbl._itbl as _itbl
import time

from kinorrt.search_space import SearchSpace
from kinorrt.mechanics.contact_kinematics import *
import random
from kinorrt.mechanics.stability_margin import *
from kinorrt.rrt import RRTManipulation
from kinorrt.mechanics.traj_optim import *

OBJECT_SHAPE = [1.75, 1, 1.5, 0.75]
HALLWAY_W = 2.5
BLOCK_H = 7
BLOCK_W = 5
np.seterr(divide='ignore')
np.set_printoptions(suppress=True, precision=4, linewidth=210)


def print_opengl_error():
    err = glGetError()
    if (err != GL_NO_ERROR):
        print('GLError: ', gluErrorString(err))

def config2trans(q):
    q = q.flatten()
    a = q[2]
    g = np.identity(3)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a), np.cos(a)]])
    g[0:2,0:2] = R
    g[0:2,-1] = q[0:2]
    return g


class iTM2d(Application):
    def __init__(self, object_shape):
        # Initialize scene.
        super(iTM2d, self).__init__(None)

        self.mesh = Box(1.0, 0.5, 0.2)
        self.light_box = Box(0.2, 0.2, 0.2)
        self.object_shape = object_shape

    def init(self):
        super(iTM2d, self).init()

        # Basic lighting shader.
        vertex_source = os.path.join(get_data(), 'shader', 'basic_lighting.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'basic_lighting.fs')
        self.basic_lighting_shader = Shader(vertex_source, fragment_source)

        # Lamp shader.
        vertex_source = os.path.join(get_data(), 'shader', 'flat.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'flat.fs')
        self.lamp_shader = Shader(vertex_source, fragment_source)

        # Normal shader.
        vertex_source = os.path.join(get_data(), 'shader', 'normals.vs')
        fragment_source = os.path.join(get_data(), 'shader', 'normals.fs')
        geometry_source = os.path.join(get_data(), 'shader', 'normals.gs')
        self.normal_shader = Shader(vertex_source, fragment_source, geometry_source)

        # Trackball camera.
        self.camera = TrackballCamera(radius=50)

        # Toggle variables.
        self.draw_mesh = True
        self.draw_wireframe = True
        self.draw_normals = False

    def init2(self):
        # C++ OpenGL.
        _itbl.loadOpenGL()

        # 2D shader.
        vertex_source = os.path.join(get_data(), 'shader', '2d.vs')
        fragment_source = os.path.join(get_data(), 'shader', '2d.fs')
        self.flat_shader = Shader(vertex_source, fragment_source)

        # Object

        self.env_contacts = None
        self.manip_contacts = None
        self.env_contacts = None
        self.manifold = None
        self.v_m = None
        self.counter = 0
        self.targets = in_hand_targets(self.object_shape)

        self.collision_manager = in_hand()


        self.all_configs_on = False
        self.step_on = False
        self.path_on = False

        self.manip_p = None
        self.next_manip_p = None

    def target_T(self,T0,T1):
        self.T0 = T0
        self.T1 = T1

    def draw_manifold(self):
        if self.manifold is None:
            return

        glPointSize(5)
        manifold = self.manifold
        for i in range(len(manifold.depths)):
            glBegin(GL_POINTS)
            cp = manifold.contact_points[i]
            glVertex3f(cp[0], cp[1], 1)
            glEnd()
            glBegin(GL_LINES)
            d = manifold.depths[i]
            n = manifold.normals[i]
            cq = cp - d * n
            glVertex3f(cp[0], cp[1], 1)
            glVertex3f(cq[0], cq[1], 1)
            glEnd()

    def draw_ground(self):
        glBegin(GL_LINES)
        # ground line
        glVertex3f(-10, 0, -1)
        glVertex3f(10, 0, -1)
        # hashes
        for x in np.arange(-10, 10, 0.1):
            glVertex3f(x, 0, -1)
            glVertex3f(x - 0.1, -0.1, -1)
        glEnd()

    def draw_grid(self, size, step):
        glBegin(GL_LINES)

        glColor3f(0.3, 0.3, 0.3)
        for i in np.arange(step, size, step):
            glVertex3f(-size, i, 0)  # lines parallel to X-axis
            glVertex3f(size, i, 0)
            glVertex3f(-size, -i, 0)  # lines parallel to X-axis
            glVertex3f(size, -i, 0)

            glVertex3f(i, -size, 0)  # lines parallel to Z-axis
            glVertex3f(i, size, 0)
            glVertex3f(-i, -size, 0)  # lines parallel to Z-axis
            glVertex3f(-i, size, 0)

        # x-axis
        glColor3f(0.5, 0, 0)
        glVertex3f(-size, 0, 0)
        glVertex3f(size, 0, 0)

        # z-axis
        glColor3f(0, 0, 0.5)
        glVertex3f(0, -size, 0)
        glVertex3f(0, size, 0)

        glEnd()

    def render(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        # glEnable(GL_CULL_FACE)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.basic_lighting_shader.use()
        print_opengl_error()

        model = glm.mat4(1.0)
        self.basic_lighting_shader.set_mat4('model', np.asarray(model))

        view = self.camera.get_view()
        self.basic_lighting_shader.set_mat4('view', np.asarray(view))

        projection = glm.perspective(glm.radians(45.0), 1200. / 900, 0.1, 100.0)
        self.basic_lighting_shader.set_mat4('projection', np.asarray(projection))

        # colors
        # self.basic_lighting_shader.set_vec3('objectColor', np.array([1.0, 0.5, 0.31], 'f'))
        self.basic_lighting_shader.set_vec3('lightColor', np.array([1.0, 1.0, 1.0], 'f'))

        # light
        lightPos = glm.vec3([1.00, 1.75, 10.0])
        self.basic_lighting_shader.set_vec3('lightPos', np.asarray(lightPos))

        # camera
        cameraPos = glm.vec3(glm.column(glm.inverse(view), 3))
        self.basic_lighting_shader.set_vec3('viewPos', np.asarray(cameraPos))

        # Draw object.
        if self.draw_mesh:
            # Draw obstacles.
            self.basic_lighting_shader.set_vec3('objectColor', get_color('gray'))
            self.collision_manager.draw(self.basic_lighting_shader.id, True, True)

            # Draw object.
            self.basic_lighting_shader.set_vec3('objectColor', get_color('clay'))
            for target in self.targets:
                target.draw3d(self.basic_lighting_shader.id)

        # Draw normals.
        self.normal_shader.use()
        self.normal_shader.set_mat4('model', np.asarray(model))
        self.normal_shader.set_mat4('view', np.asarray(view))
        self.normal_shader.set_mat4('projection', np.asarray(projection))

        if self.draw_normals:
            self.mesh.draw(self.normal_shader)

        # Draw edges and light.
        self.lamp_shader.use()
        self.lamp_shader.set_mat4('model', np.asarray(model))
        self.lamp_shader.set_mat4('view', np.asarray(view))
        self.lamp_shader.set_mat4('projection', np.asarray(projection))
        self.lamp_shader.set_vec3('objectColor', np.ones((3, 1), 'float32'))

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        if self.draw_wireframe:
            # Draw object.
            for target in self.targets:
                target.draw3d(self.lamp_shader.id)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        light_model = glm.mat4(1.0)
        light_model = glm.translate(light_model, lightPos)
        self.lamp_shader.set_mat4('model', np.asarray(light_model))
        # self.light_box.draw(self.lamp_shader)

        self.lamp_shader.set_mat4('model', np.asarray(model))
        self.lamp_shader.set_vec3('objectColor', get_color('teal'))

        model = glm.mat4(1.0)
        self.lamp_shader.set_vec3('objectColor', np.ones((3, 1), 'float32'))
        self.lamp_shader.set_mat4('model', np.asarray(model))
        # self.draw_grid(5, 0.25)

    def render2(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        # glEnable(GL_BLEND)
        # glEnable(GL_CULL_FACE)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        self.flat_shader.use()
        model = glm.mat4(1.0)
        self.flat_shader.set_mat4('model', np.asarray(model))
        view = glm.mat4(1.0)
        self.flat_shader.set_mat4('view', np.asarray(view))
        aspect_ratio = 800. / 600.
        d = 10
        ortho = glm.ortho(-d * aspect_ratio, d * aspect_ratio, -d, d, -100.0, 100.0)
        # ortho = glm.ortho(-2*aspect_ratio, 2*aspect_ratio, -2, 2, -100.0, 100.0)
        self.flat_shader.set_mat4('projection', np.asarray(ortho))
        self.flat_shader.set_vec3('offset', np.zeros((3, 1), 'float32'))
        self.flat_shader.set_float('scale', 1.0)
        self.flat_shader.set_vec3('objectColor', np.ones((3, 1), 'float32'))
        # self.draw_grid(5, 0.25)

        # Draw obstacles.
        self.flat_shader.set_vec3('objectColor', get_color('gray'))
        self.collision_manager.draw(self.flat_shader.id, True, False)

        if self.step_on:
            # Draw object.
            new_m = point_manipulator()
            if self.counter >= len(self.path):
                self.counter = 0

            self.config = self.path[self.counter]
            self.manip_p = self.mnp_path[self.counter]

            if self.manip_p is not None:
                for mnp in self.manip_p:
                    p = mnp.p
                    p = p[0:2]
                    new_m.update_config(np.array(p),self.config)
                    self.flat_shader.set_vec3('objectColor', get_color('red'))
                    new_m.obj.draw2d(self.flat_shader.id, True)

            self.flat_shader.set_vec3('objectColor', get_color('clay'))
            T2 = config2trans(np.array(self.config))
            T3 = np.identity(4)
            T3[0:2, 3] = T2[0:2, 2]
            T3[0:2, 0:2] = T2[0:2, 0:2]
            self.targets[0].transform()[:, :] = np.dot(T3, self.T0)
            self.targets[0].draw2d(self.flat_shader.id, True)
            self.targets[1].transform()[:, :] = np.dot(T3, self.T1)
            self.targets[1].draw2d(self.flat_shader.id, True)

            # print(self.counter, len(self.path))
            time.sleep(0.07)
            self.counter += 1

        if self.path_on:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            for i in range(len(self.path)):
                self.flat_shader.set_vec3('objectColor', get_color('clay'))
                target_config = self.path[i]
                T2 = config2trans(np.array(target_config))
                T3 = np.identity(4)
                T3[0:2, 3] = T2[0:2, 2]
                T3[0:2, 0:2] = T2[0:2, 0:2]
                self.targets[0].transform()[:, :] = np.dot(T3, self.T0)
                self.targets[0].draw2d(self.flat_shader.id, True)
                self.targets[1].transform()[:, :] = np.dot(T3, self.T1)
                self.targets[1].draw2d(self.flat_shader.id, True)

        if self.all_configs_on:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            # show all nodes
            for node in self.nodes:
                self.flat_shader.set_vec3('objectColor', get_color('clay'))
                target_config = np.array(node)
                T2 = config2trans(target_config)
                T3 = np.identity(4)
                T3[0:2, 3] = T2[0:2, 2]
                T3[0:2, 0:2] = T2[0:2, 0:2]
                self.targets[0].transform()[:, :] = np.dot(T3, self.T0)
                self.targets[0].draw2d(self.flat_shader.id, True)
                self.targets[1].transform()[:, :] = np.dot(T3, self.T1)
                self.targets[1].draw2d(self.flat_shader.id, True)

    def on_key_press2(self, key, scancode, action, mods):
        if key == glfw.KEY_C and action == glfw.PRESS:
            self.step_on = False
            self.path_on = False
            self.all_configs_on = False
        if key == glfw.KEY_T and action == glfw.PRESS:
            self.step_on = True

        if key == glfw.KEY_A and action == glfw.PRESS:
            self.all_configs_on = True
        if key == glfw.KEY_P and action == glfw.PRESS:
            self.path_on = True

    def on_key_press(self, key, scancode, action, mods):
        pass

    # def on_mouse_press(self, x, y, button, modifiers):
    #     pass

    # def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
    #     pass

    def on_mouse_press(self, x, y, button, modifiers):
        x = 2.0 * (x / 800.0) - 1.0
        y = 2.0 * (y / 600.0) - 1.0
        if button == 1:  # left click
            self.camera.mouse_roll(x, y, False)
        if button == 4:  # right click
            self.camera.mouse_zoom(x, y, False)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        x = 2.0 * (x / 800.0) - 1.0
        y = 2.0 * (y / 600.0) - 1.0
        if buttons == 1:  # left click
            self.camera.mouse_roll(x, y)
        if buttons == 4:  # right click
            self.camera.mouse_zoom(x, y)

    def get_path(self, path, mnp_path):
        self.path = path
        self.mnp_path = mnp_path

    def get_nodes(self, nodes):
        self.nodes = nodes
    def get_tree(self, tree):
        self.tree = tree

def in_hand():
    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(3, 0.5, 2, 0.05)
    wall2 = _itbl.Rectangle(0.2,0.8,2,0.05)

    wall1.transform()[0:3, 3] = np.array([0, 1.75, 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([0, 1.35, 0]).reshape(wall1.transform()[0:3, 3].shape)

    manager.add(wall1)
    # manager.add(wall2)


    return manager


def in_hand_targets(object_shapes):
    targets = []

    wall1 = _itbl.Rectangle(object_shapes[0][0], object_shapes[0][1], 2, 0.05)
    wall2= _itbl.Rectangle(object_shapes[1][0], object_shapes[1][1], 2, 0.05)


    wall1.transform()[0:3, 3] = np.array([0, object_shapes[0][1]/2 , 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([0, -object_shapes[1][1]/2 , 0]).reshape(wall2.transform()[0:3, 3].shape)

    targets.append(wall1)
    targets.append(wall2)

    return targets

class in_hand_part(object):
    def __init__(self, objs, object_shapes):
        self.objs = objs
        self.object_shapes = object_shapes
        self.T0 = np.copy(self.objs[0].transform())
        self.T1 = np.copy(self.objs[1].transform())

    def update_config(self, x):
        T2 = config2trans(np.array(x))
        T3 = np.identity(4)
        T3[0:2, 3] = T2[0:2, 2]
        T3[0:2, 0:2] = T2[0:2, 0:2]
        self.objs[0].transform()[:, :] = np.dot(T3,self.T0)
        self.objs[1].transform()[:, :] = np.dot(T3, self.T1)
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
        return sample_finger_contact_inhand(self.object_shapes, npts)

def sample_finger_contact_inhand(object_shapes, npts):
    w1 = object_shapes[0][0]
    l1 = object_shapes[0][1]
    w2 = object_shapes[1][0]
    l2 = object_shapes[1][1]
    contacts = []

    finger_sides = np.random.choice([0,1,2,3,4,5,6,7], npts)
    for side in finger_sides:
        if side == 0:
            n = np.array([0, -1])
            p = np.array([w1 * (np.random.random() - 0.5), l1])
        elif side == 1:
            n = np.array([-1, 0])
            p = np.array([w1/2,l1*np.random.random()])
        elif side == 2:
            n = np.array([0, 1])
            p = np.array([w2/2 + 0.5*(w1-w2)*np.random.random(), 0])
        elif side == 3:
            n = np.array([-1, 0])
            p = np.array([w2/2, -l2*np.random.random()])
        elif side == 4:
            n = np.array([0, 1])
            p = np.array([w2*(np.random.random() - 0.5), -l2])
        elif side == 5:
            n = np.array([1, 0])
            p = np.array([-w2/2, -l2*np.random.random()])
        elif side == 6:
            n = np.array([0, 1])
            p = np.array([-(w2/2 + 0.5*(w1-w2)*np.random.random()), 0])
        elif side == 7:
            n = np.array([1, 0])
            p = np.array([-w1/2,l1*np.random.random()])

        contact = Contact(p, n, 0.0)
        contacts.append(contact)
    return contacts

class in_hand_environment(object):
    def __init__(self, collision_manager):
        self.collision_manager = collision_manager

    def check_collision(self, target, target_config):
        # contacts are in the world frame
        target.update_config(target_config)
        manifold = self.collision_manager.collide(target.objs[0])
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

        manifold = self.collision_manager.collide(target.objs[1])
        if sum(np.array(manifold.depths) < 0.015) != 0:
            if_collide = True

        n_pts = len(manifold.contact_points)
        # the contacts are wrt the object frame
        for i in range(n_pts):
            if manifold.depths[i] >= 0.015:
                continue
            cp = manifold.contact_points[i]
            cn = manifold.normals[i]
            ci = Contact(cp, cn, manifold.depths[i])
            contacts.append(ci)

        return if_collide, contacts

def test_kinorrt_cases(stability_solver, max_samples = 100):

    viewer = Viewer()
    _itbl.loadOpenGL()

    step_length = 2
    neighbor_r = 5
    dist_cost = 10

    object_shapes = [[1,0.5],[0.5,0.5]]
    X_dimensions = np.array([(-1.5, 1.5), (-1.5, 2), (-1.5*np.pi, 1.5*np.pi)])
    x_init = (0,0,0)
    x_goal = (0,0,np.pi)
    world_key = 'vert'
    dist_weight = 1
    manipulator = doublepoint_manipulator(np.array([[-1.5,-1.5,0.,-1.5],[-0.,1.5,1.5,1.5]]))
    mnp_fn_max = 100
    goal_kch = [0.1, 0.1, 1]

    app = iTM2d(object_shapes)
    viewer.set_renderer(app)
    viewer.init()

    X = SearchSpace(X_dimensions)

    the_object = in_hand_part(app.targets,object_shapes)

    app.target_T(the_object.T0, the_object.T1)
    envir = in_hand_environment(app.collision_manager)
    rrt_tree = RRTManipulation(X, x_init, x_goal, envir, the_object, manipulator,
                                max_samples, neighbor_r, world_key)
    rrt_tree.env_mu = 0.8
    rrt_tree.mnp_mu = 0.8
    rrt_tree.mnp_fn_max = mnp_fn_max
    rrt_tree.dist_weight = dist_weight
    rrt_tree.cost_weight[0] = dist_cost
    rrt_tree.step_length = step_length
    rrt_tree.goal_kch = goal_kch

    rrt_tree.initialize_stability_margin_solver(stability_solver)

    t_start = time.time()

    init_mnp = [Contact((-0.5,0.25),(1,0),0),Contact((0.5,0.25),(-1,0),0)]
    # rrt_tree.x_goal = (0,0,np.pi/2)
    # path, mnp_path = rrt_tree.search(init_mnp)
    rrt_tree.x_goal = (0, 0, np.pi)
    paths = rrt_tree.search(init_mnp)

    t_end = time.time()
    print('time:', t_end - t_start)

    whole_path = []
    envs = []
    mnps = []
    modes = []
    for q in paths[2][2:]:
        ps = rrt_tree.trees[0].edges[q].path
        ps.reverse()
        m = np.array(rrt_tree.trees[0].edges[q].mode)
        current_envs = []
        current_modes = []
        current_path = []
        mnp = rrt_tree.trees[0].edges[q].manip
        for p in ps:
            _, env = rrt_tree.check_collision(p)
            if len(mnp) + len(env) != len(m):
                if len(mnp) + len(env) == sum(m != CONTACT_MODE.LIFT_OFF):
                    m = m[m != CONTACT_MODE.LIFT_OFF]
                else:
                    print('env contact error')
                    continue
            current_modes.append(m)
            current_path.append(p)
            current_envs.append(env)

        current_mnps = [mnp] * len(current_path)
        whole_path += current_path
        envs += current_envs
        modes += current_modes
        mnps += current_mnps

    print(whole_path, envs, modes, mnps)
    # app.get_path(paths[0],mnps)
    results = traj_optim_static((whole_path, envs, modes, mnps), rrt_tree)

    app.get_path(np.array(results).reshape(-1, 3), mnps)
    app.get_nodes(rrt_tree.trees[0].nodes)
    app.get_tree(rrt_tree)
    viewer.start()

    return

stability_solver = StabilityMarginSolver()
for i in [4]:
    seed_number = i*1000
    random.seed(seed_number)
    np.random.seed(seed_number)
    ti = test_kinorrt_cases(stability_solver, max_samples=1000)

