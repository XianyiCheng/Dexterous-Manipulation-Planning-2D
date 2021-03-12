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
from kinorrt.rrt import RRTManipulation

import plotly.graph_objects as go

class iTM2d(Application):
    def __init__(self, object_shape, example='sofa'):
        # Initialize scene.
        super(iTM2d, self).__init__(None)

        self.mesh = Box(1.0, 0.5, 0.2)
        self.light_box = Box(0.2, 0.2, 0.2)
        self.example = example
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
        self.target = _itbl.Rectangle(self.object_shape[0] * 2, self.object_shape[1] * 2, 2, 0.0)
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
            self.collision_manager = corner()
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
        else:
            print('Cannot find collision manager!')
            raise

        self.all_configs_on = False
        self.step_on = False
        self.path_on = False

        self.manip_p = None
        self.next_manip_p = None

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
            self.target.draw3d(self.basic_lighting_shader.id)

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
            self.target.draw3d(self.lamp_shader.id)

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
            self.target.transform()[:, :] = T3
            self.target.draw2d(self.flat_shader.id, True)

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
                self.target.transform()[:, :] = T3
                self.target.draw2d(self.flat_shader.id, True)

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
                self.target.transform()[:, :] = T3
                self.target.draw2d(self.flat_shader.id, True)

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


object_shape = [1, 0.2, 0.2, 0.2]
X_dimensions = np.array([(-4.5, 4.5), (2, 3.5), (-2 * np.pi, 2 * np.pi)])
x_init = (0, 2.2, 0)
x_goal = (0, 3, -np.pi / 2)
world_key = 'vert'
dist_weight = 1

mnp_fn_max = 6
goal_kch = [0.01, 0.1, 1]
allow_contact_edges = [True, False, True, False]

viewer = Viewer()
_itbl.loadOpenGL()
manipulator = doublepoint_manipulator()
mnp_fn_max = None
step_length = 2
neighbor_r = 5
dist_cost = 1

app = iTM2d(object_shape, example='book')
viewer.set_renderer(app)
viewer.init()

X = SearchSpace(X_dimensions)

the_object = part(app.target, object_shape, allow_contact_edges)

rrt_tree = RRTManipulation(X, x_init, x_goal, environment(app.collision_manager), the_object, manipulator,
                           50, neighbor_r, world_key)

x = (-2.3, 2.2, 0)
x_rand = (-2.5, 2.2, -np.pi)
x_rand1 = (-3,3.5,-np.pi/3)
_, envs = rrt_tree.check_collision(x)
mnps = [Contact((-0.8,0.2),(0,-1),0),Contact((-0.8,-0.2),(0,1),0)]
mode = [CONTACT_MODE.FOLLOWING,CONTACT_MODE.FOLLOWING,CONTACT_MODE.SLIDING_LEFT,CONTACT_MODE.LIFT_OFF]
x_new, path, _ = rrt_tree.forward_integration(x,x_rand,envs,mnps,mode)
path += [(-2.432,3,-np.pi/2)]
x_new, path1, _ = rrt_tree.forward_integration(x,x_rand1,envs,mnps,mode)
path += [(-2.432,3,-np.pi/2)]
print(x_new)
print(path)

fig = go.Figure()
boundary = []
for theta in np.arange(0,np.pi/2,0.1):
    q = (-3-1*np.cos(theta)+0.2*np.sin(theta), 2+0.2*np.cos(theta)+1*np.sin(theta),-theta)
    boundary += [q]
boundary += [(-2.8,3,-np.pi/2)]

for x0 in np.arange(-2.8,-1.1,0.1):
    boundary += [(x0,3,-np.pi/2)]
boundary += [(-1.1,3,-np.pi/2)]

b0 = []
for theta in np.arange(0,np.pi/2,0.1):
    q = (-1.3-1*np.cos(theta)+0.2*np.sin(theta), 2+0.2*np.cos(theta)+1*np.sin(theta),-theta)
    b0 += [q]

b0.reverse()
boundary+=b0

for x0 in np.arange(-4,-2.3,0.1):
    boundary += [(x0,2.2,0)]



'''
x1, path1, _ = rrt_tree.forward_integration(x,(-4,2.2,0),envs,mnps,mode)
x2, path2, _ = rrt_tree.forward_integration(x,(-4,2.2,-np.pi/4),envs,mnps,mode)
x3, path3, _ = rrt_tree.forward_integration(x,(-3,2.6,-np.pi/3),envs,mnps,mode)
x4, path4, _ = rrt_tree.forward_integration(x,(-4,3,-np.pi),envs,mnps,mode)
x5, path5, _ = rrt_tree.forward_integration(x,(-0.5,3,-np.pi),envs,mnps,mode)
x6, path6, _ = rrt_tree.forward_integration(x,(-4,3,-np.pi/2.5),envs,mnps,mode)
x7, path7, _ = rrt_tree.forward_integration(x,(-2,3,-np.pi),envs,mnps,mode)
'''
xb, yb, zb = np.array(boundary).T
#xs, ys, zs = np.array(boundary + path+path1+path2+path3+path4+path5+path6+path7).T


x,y,z = np.array(path).T
x1,y1,z1 = np.array(path1).T
fig = go.Figure()

fig.add_trace(go.Scatter3d(x=xb, y=zb, z=yb, mode='lines', line={'width':8, 'color':'blue'},name='Manifold Boundary'))
# fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', opacity=0.50))
fig.add_trace(go.Scatter3d(x=x, y=z, z=y, name='trajectory 1', mode='lines+markers',line={'width':4,'color':'green'},marker={'size':4,'color':'green'}))
fig.add_trace(go.Scatter3d(x=x1, y=z1, z=y1,name='trajectory 2', mode='lines+markers',line={'width':4,'color':'red'}, marker={'size':4,'color':'red'}))
fig.add_trace(go.Scatter3d(x=[x_rand[0]], y=[x_rand[2]], z=[x_rand[1]], mode='markers',name = 'goal 1', marker={'size':6,'color':'green'}))
fig.add_trace(go.Scatter3d(x=[x_rand1[0]], y=[x_rand1[2]], z=[x_rand1[1]],mode='markers',name = 'goal 2', marker={'size':6,'color':'red'}))
fig.update_layout(
    scene={
        'xaxis_title':'x',
        'yaxis_title' : 'θ',
        'zaxis_title' : 'y',
        'aspectmode': 'cube'
    })
#fig.write_html("./forward_integration.html")
fig.show()


