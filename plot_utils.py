from matplotlib import animation
import meshzoo
import optimesh
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
import keras.backend as K
import pyvista as pv
import os

#os.system('/usr/bin/Xvfb :99 -screen 0 1024x768x24 &')
#os.environ['DISPLAY'] = ':99'

def image_plot(graphs,a,b):

    points = graphs.x
    cells_ = graphs.cells

    cells = np.concatenate((np.full((len(cells_), 1), 3), cells_), axis=1)
    mesh = pv.PolyData(points,cells)

    display_args = dict(show_edges=True, color=True)

    p = pv.Plotter(shape=(3, 3))

    for i in range(3):
        p.subplot(i, 0)
        p.add_mesh(mesh, **display_args)
        p.add_text("Original Mesh")

    def row_plot(row, subfilter):
        subs = [a, b]
        for i in range(2):
            p.subplot(row, i + 1)
            p.add_mesh(mesh.subdivide(subs[i], subfilter=subfilter), **display_args)
            p.add_text(f"{subfilter} subdivision of {subs[i]}")

    row_plot(0, "linear")
    row_plot(1, "butterfly")
    row_plot(2, "loop")

    # cpos = [
    #     (-0.02788175062966399, 0.19293295656233056, 0.4334449972621349),
    #     (-0.053260899930287015, 0.08881197167521734, -9.016948161029588e-05),
    #     (-0.10170607813337212, 0.9686438023715356, -0.22668272496584665),
    # ]

    p.link_views()
    p.view_isometric()
    # p.camera_position = cpos
    p.show(screenshot='test.png')
    return p


def roatation_animation(graphs,stress,animation_save_name):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    stress_check = np.random.rand(len(graphs.x))

    def init():
        points = graphs.x
        cells = graphs.cells

        ax.plot_trisurf(points[:, 0], points[:, 1], cells, points[:, 2],facecolors=stress,cmap ='jet',shade=True)
        return fig,

    def animate(i):
        ax.view_init(elev=30., azim=i)
        # axs[1].view_init(elev=30., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=list(range(0, 360, 10)),
                                    interval=20, blit=True)  # ,repeat=False)
    anim.save(f"{animation_save_name}_rotation.gif", writer='imagemagick', fps=5)


def remesh_animation(graphs,stress,animation_save_name):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    points = graphs.x
    cells = graphs.cells

    ### test
    # points, cells= graphs

    # mesh = meshplex.MeshTri(points, cells)
    points, cells, points_set = optimesh.optimize_points_cells(
        points,
        cells,
        "CVT (full)",
        1.0e-2,
        100,
        verbose=False,
        # implicit_surface=Sphere(),
        # step_filename_format="out{:03d}.vtk",
        points_set_bool=True
    )

    def animate(i):
        ax.plot_trisurf(points_set[i][:, 0], points_set[i][:, 1], cells, points_set[i][:, 2], facecolors=stress, cmap='jet', shade=True)
        # ax.view_init(elev=30., azim=i)
        # ax.plot_trisurf(points[:, 0], points[:, 1], cells, points[:, 2], facecolors=stress, cmap='jet',shade=True)
        # axs[1].view_init(elev=30., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=range(5),
                                   interval=20, blit=True)  # ,repeat=False)
    # anim = animation.FuncAnimation(fig, animate, frames=range(0,len(points_set),10),
    #                                 interval=20, blit=True)  # ,repeat=False)
    anim.save(f"{animation_save_name}_remesh.gif", writer='imagemagick', fps=5)

# def remesh_animation(graphs,stress):
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#
#     # points = graphs.x
#     # cells = graphs.cells
#
#     ### test
#     points, cells= graphs
#
#     # mesh = meshplex.MeshTri(points, cells)
#     points, cells, points_set = optimesh.optimize_points_cells(
#         points,
#         cells,
#         "CVT (full)",
#         1.0e-2,
#         100,
#         verbose=False,
#         # implicit_surface=Sphere(),
#         # step_filename_format="out{:03d}.vtk",
#         points_set_bool=True
#     )
#
#     def animate(i):
#         ax.plot_trisurf(points_set[i][:, 0], points_set[i][:, 1], cells, points_set[i][:, 2], facecolors=stress, cmap='jet', shade=True)
#         # ax.view_init(elev=30., azim=i)
#         # axs[1].view_init(elev=30., azim=i)
#         return fig,
#
#     anim = animation.FuncAnimation(fig, animate, frames=range(0,len(points_set),10),
#                                     interval=20, blit=True)  # ,repeat=False)
#     anim.save("remesh_2.gif", writer='imagemagick', fps=5)

class Sphere:
    def f(self, x):
        return 1.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    def grad(self, x):
        return -2 * x

def remesh_animation_test():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    points, cells = meshzoo.tetra_sphere(20)

    stress = np.random.rand(len(points))

    points, cells, points_set = optimesh.optimize_points_cells(
        points,
        cells,
        "CVT (full)",
        1.0e-5,
        100,
        verbose=False,
        implicit_surface=Sphere(),
        # step_filename_format="out{:03d}.vtk",
        points_set_bool=True
    )

    def animate(i):
        ax.plot_trisurf(points_set[i][:, 0], points_set[i][:, 1], cells, points_set[i][:, 2], facecolors=stress, cmap='jet', shade=True)
        # ax.view_init(elev=30., azim=i)
        # ax.plot_trisurf(points[:, 0], points[:, 1], cells, points[:, 2], facecolors=stress, cmap='jet',shade=True)
        # axs[1].view_init(elev=30., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames = list(range(0, len(points_set), 10)),
                                   interval=20, blit=True)  # ,repeat=False)
    # anim = animation.FuncAnimation(fig, animate, frames=range(0,len(points_set),10),
    #                                 interval=20, blit=True)  # ,repeat=False)
    anim.save("remesh_7.gif", writer='imagemagick', fps=5)

def grad_cam(dataset_te,pre_model):

    inputs = (dataset_te[0].x[np.newaxis, :, :],dataset_te[0].a[np.newaxis, :, :])

    cam = CAM(pre_model)
    mask = cam.getMasks(inputs)
    mask = np.array(mask)
    # mask_ = mask[:graph_for_plot.n_nodes,]
    # mask_ = mask[:,0]# print('original_mask:',mask)
    # mask /= mask.max()
    # print('normalized_mask:', mask)
    # masks_c0, masks_c1 = mask
    # print(masks_c0)
    # print(masks_c1)

class CAM:
    def __init__(self, model):
        self.weights = K.eval(model.layers[-1].weights[0])
        self.bias = K.eval(model.layers[-1].weights[1])
        self.tempModel = Model(model.input, model.layers[-4].output)
        self.getMasks = self.getCAM

    def getCAM(self, XY):
        temp = np.matmul(self.tempModel.predict(XY), self.weights).squeeze() + self.bias
        return (1*(temp>0)*temp)
