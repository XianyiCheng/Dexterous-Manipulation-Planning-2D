import numpy as np

def adjointTrans_2d(g):
    adg = np.identity(3)
    R = g[0:2,0:2]
    p = g[0:2,-1]

    adg[0:2,0:2] = R
    adg[0:2,-1] = np.array([p[1],-p[0]])

    #adg = [R,[t2;-t1];0, 1];
    return adg

def inv_g_2d(g):

    g_inv = np.identity(3)
    g_inv[0:2,0:2] = g[0:2,0:2].T
    g_inv[0:2,-1] = -np.dot(g[0:2,0:2].T,g[0:2,-1])

    # g_inv = [R.T, -R.T*t; 0, 1] g = [R,t;0,1]
    return g_inv

def compute_transformation(p,n):
    # print(p)
    # print(n)
    p = np.array(p)
    n = np.array(n)
    g = np.identity(3)
    R = np.array([[n[1],n[0]], [-n[0],n[1]]])
    g[0:2,0:2] = R
    g[0:2,-1] = p
    return g

def config2trans(q):
    q = np.array(q)
    a = q[2]
    g = np.identity(3)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a), np.cos(a)]])
    g[0:2,0:2] = R
    g[0:2,-1] = q[0:2]
    return g

def trans2config(g):
    q = np.zeros(3)
    q[0:2] = g[0:2,-1]
    q[2] = np.acos(g[0,0])
    return q
