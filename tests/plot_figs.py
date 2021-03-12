import numpy as np
import plotly.graph_objects as go

def sample_manifold():
    half_w =  1
    half_l = 2
    thetas = np.arange(np.pi/4,np.pi/2,0.01)
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=thetas, mode='lines',line={'width':10}))
    fig.add_trace(go.Scatter3d(x=[np.cos(np.pi/2),np.cos(np.pi/4)], y=[np.sin(np.pi/2),np.sin(np.pi/4)], z=[np.pi/2,np.pi/4], mode='markers',marker={'size':5,'color':'blue'}))
    fig.add_trace(go.Scatter3d(x=[0.2], y=[1.2], z=[np.pi/2-np.pi/10], mode='markers'))
    fig.update_layout(scene_aspectmode='cube')
    fig.show()

sample_manifold()