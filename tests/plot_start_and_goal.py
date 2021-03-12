
from time import time

import glm
from itbl import Ray, Shader
from itbl.accelerators import SDF, BVHAccel
from itbl.cameras import TrackballCamera

from itbl.shapes import Box
from itbl.util import get_color, get_data
from itbl.viewer import Application, Viewer
from itbl.viewer.backend import *
from wilson import *
import itbl._itbl as _itbl
import time

from kinorrt.search_space import SearchSpace
from kinorrt.mechanics.contact_kinematics import *
import random
from kinorrt.mechanics.stability_margin import *

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
    def __init__(self, object_shape, example, x_start, x_goal):
        # Initialize scene.
        super(iTM2d, self).__init__(None)

        self.mesh = Box(1.0, 0.5, 0.2)
        self.light_box = Box(0.2, 0.2, 0.2)
        self.example = example
        self.object_shape = object_shape
        self.x_start = x_start
        self.x_goal = x_goal

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

        if self.example == 'sofa':
            self.collision_manager = create_hallway(HALLWAY_W, BLOCK_W, BLOCK_H, self.object_shape[
                0] * 2.5 + BLOCK_W * 0.5)  # uniform_sample_maze((4,4), 3, 1.25)
        elif self.example == 'maze':
            self.collision_manager = uniform_sample_maze((3, 3), 3, 1.25)
        elif self.example == 'corner':
            self.collision_manager = corner()
        elif self.example == 'wall':
            self.collision_manager = wall()
        elif self.example == 'table':
            self.collision_manager = table()
        elif self.example == 'obstacle_course':
            self.collision_manager = obstacle_course()
        elif self.example == 'peg-in-hole-v':
            self.collision_manager = peg_in_hole_v()
        elif self.example == 'peg-in-hole-p':
            self.collision_manager = peg_in_hole_p()
        elif self.example == 'book':
            self.collision_manager = book()
        elif self.example == 'unpacking':
            self.collision_manager = unpacking()
        elif self.example == 'pushing':
            self.collision_manager = pushing()
        elif self.example == 'inhand':
            self.collision_manager = in_hand()
            self.targets = in_hand_targets(self.object_shape)
        else:
            print('Cannot find collision manager!')
            raise
        if self.example != 'inhand':
            self.target = _itbl.Rectangle(self.object_shape[0] * 2, self.object_shape[1] * 2, 2, 0.0)

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

        if self.example == 'inhand':
            # self.flat_shader.set_vec3('objectColor', get_color('green'))
            # config = self.x_start
            # T2 = config2trans(np.array(config))
            # T3 = np.identity(4)
            # T3[0:2, 3] = T2[0:2, 2]
            # T3[0:2, 0:2] = T2[0:2, 0:2]
            # self.targets[0].transform()[:, :] = np.dot(T3, self.T0)
            # self.targets[0].draw2d(self.flat_shader.id, True)
            # self.targets[1].transform()[:, :] = np.dot(T3, self.T1)
            # self.targets[1].draw2d(self.flat_shader.id, True)

            self.flat_shader.set_vec3('objectColor', get_color('red'))
            config = self.x_goal
            T2 = config2trans(np.array(config))
            T3 = np.identity(4)
            T3[0:2, 3] = T2[0:2, 2]
            T3[0:2, 0:2] = T2[0:2, 0:2]
            self.targets[0].transform()[:, :] = np.dot(T3, self.T0)
            self.targets[0].draw2d(self.flat_shader.id, True)
            self.targets[1].transform()[:, :] = np.dot(T3, self.T1)
            self.targets[1].draw2d(self.flat_shader.id, True)
        else:
            self.flat_shader.set_vec3('objectColor', get_color('green'))
            target_config = self.x_start
            T2 = config2trans(np.array(target_config))
            T3 = np.identity(4)
            T3[0:2, 3] = T2[0:2, 2]
            T3[0:2, 0:2] = T2[0:2, 0:2]
            self.target.transform()[:, :] = T3
            self.target.draw2d(self.flat_shader.id, True)

            self.flat_shader.set_vec3('objectColor', get_color('red'))
            target_config = self.x_goal
            T2 = config2trans(np.array(target_config))
            T3 = np.identity(4)
            T3[0:2, 3] = T2[0:2, 2]
            T3[0:2, 0:2] = T2[0:2, 0:2]
            self.target.transform()[:, :] = T3
            self.target.draw2d(self.flat_shader.id, True)



    def on_key_press2(self, key, scancode, action, mods):
        pass

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

def visualize_cases(keyword):

    viewer = Viewer()
    _itbl.loadOpenGL()
    manipulator = point_manipulator()
    mnp_fn_max = None
    step_length = 2
    neighbor_r = 5
    dist_cost = 1


    if keyword == 'corner':
        neighbor_r = 5
        object_shape = [0.5, 0.5, 0.2, 0.2]
        X_dimensions = np.array([(0, 4), (0, 4), (-np.pi, np.pi)])
        x_init = (3, 0.5, 0)  # starting location
        x_goal = (0.707, 0.707, -np.pi / 4)
        # x_init = (2, 0.5, 0)
        # x_goal = (-3, 0.5, -np.pi / 2)
        world_key = 'vert'
        dist_weight = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 6
        step_length = 2
        goal_kch = [1, 1, 1]
    elif keyword == 'wall':
        neighbor_r = 10
        object_shape = [0.5,0.5,0.2,0.2]
        X_dimensions = np.array([(-8, 8), (0, 7 + object_shape[1]*4), (-np.pi, np.pi)])
        x_init = (4.5, 0.5, 0)
        x_goal = (-3, 7.5, 0)
        world_key = 'vert'
        dist_weight = 1
        dist_cost = 0.2
        manipulator = point_manipulator()
        step_length = 10
        mnp_fn_max = 50
        goal_kch = [1,1,0]
    elif keyword == 'table':

        object_shape = [1, 1, 0.2, 0.2]
        X_dimensions = np.array([(0, 4), (0, 4), (-2*np.pi, 2*np.pi)])
        # x_init = (3, 0.5, 0)  # starting location
        # x_goal = (0.707, 0.707, np.pi / 4)
        x_init = (2, 1, 0)
        x_goal = (2, 1, -np.pi/2)
        world_key = 'vert'
        dist_weight = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 6
        step_length = 3.14
        goal_kch = [0.1, 0.1, 10]

    elif keyword == 'obstacle_course':
        object_shape = [0.5, 0.5, 0.2, 0.2]
        X_dimensions = np.array([(-2.5,3), (0, 4), (-2*np.pi, 2*np.pi)])
        x_init = (-2.5, 1.5, 0)
        x_goal = (2.5, 1.5, 0)
        world_key = 'vert'
        dist_weight = 0.08
        dist_cost = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 6.15
        goal_kch = [0.7,0.2,0]
    elif keyword == 'peg-in-hole-v':
        object_shape = [0.45, 1, 0.2, 0.2]
        X_dimensions = np.array([(-2, 1), (-2, 3), (-np.pi, np.pi)])
        x_init = (-2,1,0)
        x_goal = (0,-1,0)
        world_key = 'vert'
        dist_weight = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 50
        goal_kch = [0.5, 0.2, 0.8]
        init_mnp = [Contact((-0.45, 0.8), (1, 0), 0), Contact((0.45, 0.8), (-1, 0), 0)]

    elif keyword == 'peg-in-hole-p':
        object_shape = [0.45, 1, 0.2, 0.2]
        X_dimensions = np.array([(-2, 3), (0,2.5), (-np.pi, np.pi)])
        x_init = (3,2.5,0)
        x_goal = (-1,0.5,np.pi/2)
        world_key = 'planar'
        dist_weight = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 50
        goal_kch = [1, 1, 1]


    elif keyword == 'unpacking':
        object_shape = [0.39, 1, 0.2, 0.2]
        X_dimensions = np.array([(-2, 2), (-0.5, 2.5), (-np.pi, np.pi)])
        x_init = (-0.5,0,0)
        x_goal = (1,1.39,-np.pi/2)
        world_key = 'vert'
        dist_weight = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 100
        goal_kch = [1, 1, 1]

    elif keyword == 'book':
        object_shape = [1, 0.2, 0.2, 0.2]
        X_dimensions = np.array([(-4.5, 4.5), (2, 3.5), (-2*np.pi, 2*np.pi)])
        x_init = (0,2.2,0)
        x_goal = (-2,3,-np.pi/2)
        world_key = 'vert'
        dist_weight = 1
        manipulator = doublepoint_manipulator()
        mnp_fn_max = 15
        goal_kch = [0.01, 0.1, 1]
        allow_contact_edges = [True, False, True, False]

    elif keyword == 'pushing':
        object_shape = [0.5, 1, 0.2, 0.2]
        X_dimensions = np.array([(-3.1, 3.1), (-2.6,4), (-np.pi, np.pi)])
        x_init = (-2,-1.25,0)
        x_goal = (2.1,2.75,0)
        world_key = 'planar'
        dist_weight = 1
        manipulator = point_manipulator()
        mnp_fn_max = 50
        goal_kch = [1, 1, 1]
    elif keyword == 'inhand':
        object_shape = [[1, 0.5], [0.5, 0.5]]
        X_dimensions = np.array([(-1.5, 1.5), (-1.5, 2), (-1.5 * np.pi, 1.5 * np.pi)])
        x_init = (0, 0, 0)
        x_goal = (0, 0, np.pi)

    else:
        print('Wrong case keyword!')
        raise


    app = iTM2d(object_shape, keyword, x_init,x_goal)
    viewer.set_renderer(app)
    viewer.init()
    if keyword == 'inhand':
        the_object = in_hand_part(app.targets, object_shape)
        app.target_T(the_object.T0, the_object.T1)


    viewer.start()

    return

ti = visualize_cases('inhand')

