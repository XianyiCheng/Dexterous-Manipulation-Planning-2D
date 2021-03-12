#from rtree import index


class Tree(object):
    def __init__(self, X):
        """
        Tree representation
        :param X: Search Space
        """
        p = index.Property()
        p.dimension = X.dimensions
        self.V = index.Index(interleaved=True, properties=p)  # vertices in an rtree
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent
        self.manip = {}
        self.env = {}
        self.path = {}
        self.nodes = []

    def add(self, X, X_parent, mnps=None, envs = None, path=None):
        self.E[X] = X_parent
        self.manip[X] = mnps
        self.env[X] = envs
        self.path[X] = path
        self.nodes.append(X)
        self.V.insert(0, X + X, X)
        self.V_count += 1

class Tree_w_modes(Tree):
    def __init__(self, X):
        super().__init__(X)
        self.modes = {} # the mode that connect to this status
        self.enum_modes = {} # {(x):['','','']} list of modes

    def add(self, X, X_parent, mnps=None, envs = None, path=None, mode = None):
        self.E[X] = X_parent
        self.manip[X] = mnps
        self.env[X] = envs
        self.path[X] = path
        self.modes[X] = mode
        self.nodes.append(X)
        self.V.insert(0, X + X, X)
        self.V_count += 1

    def add_mode_enum(self, X, modes):
        self.enum_modes[X] = modes

class RRTTree(object):
    def __init__(self):
        self.nodes = []
        self.goal_expand = {}
        self.edges = {}
        self.enum_modes = {}
        self.costs = {}

    def add(self, X, edge):
        self.nodes.append(X)
        self.goal_expand[X] = False
        self.edges[X] = edge

    def add_mode_enum(self, X, modes):
        self.enum_modes[X] = modes

class RRTEdge(object):
    def __init__(self, parent, manip, env, path, mode, manip_config = None, manipulator_collide = False):
        self.parent = parent
        self.manip = manip
        self.manip_config = None
        self.env = env
        self.path = path
        self.mode = mode
        self.manipulator_collide = manipulator_collide


