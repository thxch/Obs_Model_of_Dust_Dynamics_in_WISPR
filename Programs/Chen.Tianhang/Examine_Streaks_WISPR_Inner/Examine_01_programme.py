"""
This is a script for streaks examination on what it is, impact-generated debris or primary interplanetary dust particles.
The 01 code is based on the projective transformation theory(vanishing point).
This examination is just geometry analysis. Optics is not considered.
By 2022.3.24, Tianhang Chen
The INPUT file is loaded in 'Data' folder including the stl file and the FITS file.
There is no explicit OUTPUT file in this programme, while a 'Plotly' 3d figure will be loaded in your browser. 
But the screenshots(png) are uploaded in 'Data' folder, providing a general result of the analysis.
"""
import spiceypy as spice
from sunpy.net import attrs as a
import sunpy.map
import sunpy.io.fits
from sunpy.net import Fido
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.time import TimeDelta
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm
import copy
import os
from datetime import datetime
import sklearn.cluster as cluster
###########################################
# all codes about the package 'plotly' refer to the programme below.
# @Filename: Plot_Spacecraft.py
# @Aim: load an stl file and visualize it with appropriate scaling and rotation.
# @Author: Ziqi Wu
# @Date of Last Change: 2022-02-18
# #########################################
import pyvista as pv
from plotly import graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots

import furnsh_all_kernels      # all spice kernels needed are described in this file.

AU = 1.496e8  # Now that the analysis is near PSP spacecraft, this constant seems useless
streak_number = 8  # The numbers of selected streaks in the WISPR_Inner at observing time.
fits_file_path = '/FITS_FILE_PATH'
psp_3d_model_path = '/3D_MODEL_PATH'
WISPR_pos = np.array([0.855, -0.249, -0.300], dtype=float)  # the position of WISPR onboard PSP in spacecraft frame.
line_flag = False
# line_flag being True indicates the streaks in WISPR_INNER are parallel to each other,
# while False represents they intercept at one common point.
# The two cases need different algorithm to derive streaks' orientation/position in 3d world frame.


def add_psp_3d_model(pos, rot_theta, scale=float(10)):
    """
    :param pos: A vector [x,y,z]. Center position of the spacecraft.
    :param rot_theta: A float (deg). Rotation around the z axis centered at pos.
    :param scale: A float, 10 by default. Scaling of the spacecraft.
    :return: A trace (go.Mesh3d()) for plotly.
    Note that the z axis in stl file is the true -z axis, x axis is the true y axis and y axis is the true -x axis!
    ——Tianhang Chen, 2022.3.18
    """
    mesh = pv.read(psp_3d_model_path)
    # mesh.plot()    showed by pyvista package.
    # scale = 10
    mesh.points = scale * mesh.points / np.max(mesh.points)
    # theta_x = 80
    # theta_z = -90 + rot_theta
    # theta_y = 0
    # axes = pv.Axes(show_actor=True, actor_scale=5.0, line_width=10)
    # axes.origin = (0, 0, 0)
    # rot = mesh.rotate_x(theta_x, point=axes.origin, inplace=False)
    # rot = rot.rotate_z(theta_z, point=axes.origin, inplace=False)

    # # Visualize by Pyvista
    # p = pv.Plotter()
    # p.add_actor(axes.actor)
    # p.add_mesh(rot)
    # # p.add_mesh(mesh)
    # p.show()

    vertices = mesh.points
    triangles = mesh.faces.reshape(-1, 4)
    trace = go.Mesh3d(x=vertices[:, 1] + pos[1], y=-(vertices[:, 0] + pos[0]), z=-(vertices[:, 2] + pos[2]),
                      opacity=1,            # the inner configuration of PSP is not significant, thus opacity set 1;
                      color='silver',       # the color changes from gold to silver,
                                            # in order to emphasize the dust traces and FOV   ——Tianhang Chen, 2022.3.18
                      i=triangles[:, 1], j=triangles[:, 2], k=triangles[:, 3],
                      showscale=False,
                      )
    return trace


def get_psp_state(timestr, the_format='%Y%m%dT%H%M%S'):
    """
    :param timestr: the string showing the time. Format of it is '%Y%m%dT%H%M%S'
    :param the_format: the format of the timestr
    :return: etime: epoch time(SPICE)
             rotation matrix: the rotation(transformation) matrix from WISPR-I frame to PSP spacecraft frame.
    The definition of the two frames is described in frame kernel spp_v300.tf. (https://sppgway.jhuapl.edu/psp_frames)
    """
    etime = spice.datetime2et(datetime.strptime(timestr, the_format))
    z_unit_vec_sc = [0, 0, 1e-3]
    z_unit_vec_sc = np.array(z_unit_vec_sc, dtype=float)
    y_unit_vec_sc = [0, 1e-3, 0]
    y_unit_vec_sc = np.array(y_unit_vec_sc, dtype=float)
    x_unit_vec_sc = [1e-3, 0, 0]
    x_unit_vec_sc = np.array(x_unit_vec_sc, dtype=float)
    # z_spacecraft, _ = spice.spkcpt(z_unit_vec_sc, 'SPP', 'SPP_SPACECRAFT', etime,
    #                                'SPP_HCI', 'OBSERVER', 'NONE', 'SPP')
    z_WISPR_I, _ = spice.spkcpt(z_unit_vec_sc, 'SPP', 'SPP_WISPR_INNER', etime,
                                'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    y_WISPR_I, _ = spice.spkcpt(y_unit_vec_sc, 'SPP', 'SPP_WISPR_INNER', etime,
                                'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    x_WISPR_I, _ = spice.spkcpt(x_unit_vec_sc, 'SPP', 'SPP_WISPR_INNER', etime,
                                'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    x_unit_vec_WISPR_I = (x_WISPR_I[0:3]) * 1e3
    y_unit_vec_WISPR_I = (y_WISPR_I[0:3]) * 1e3
    z_unit_vec_WISPR_I = (z_WISPR_I[0:3]) * 1e3   # 1e3 is just a scaling factor
    rotation_matrix = np.zeros([3, 3], dtype='float')
    rotation_matrix[:, 0] = x_unit_vec_WISPR_I[:]
    rotation_matrix[:, 1] = y_unit_vec_WISPR_I[:]
    rotation_matrix[:, 2] = z_unit_vec_WISPR_I[:]
    # my_matrix = spice.sxform('SPP_HCI', 'SPP_WISPR_INNER', etime)
    # Code above is another method to derive transforming matrix via spice.sxform().
    # The function 'sxform()' returns a 6*6 matrix，including the 1st order differential of positions(velocities).
    # Since the whole 3d model of psp is imported, there is no need to introduce PSP_HCI frame, though it's still used
    # in some places in this code.
    return etime, rotation_matrix


def retrieve_streaks(num_streaks):
    """
    :param num_streaks: (integer) numbers of streaks
    :return: the slopes and intercepts of every streaks in WISPR-INNER pixel frame.(n * 2 ndarrays, the second index
    0 means the slope, 1 means intercept)
    The tool of show this fig: matplotlib.
    """
    my_fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111)
    norm_SL = colors.SymLogNorm(linthresh=0.001 * 1e-10, linscale=0.1 * 1e-10, vmin=-0.0038 * 1e-10, vmax=0.14 * 1e-10)
    plt.imshow(true_data, cmap=my_colormap, norm=norm_SL)
    ax.set_xlabel('x-axis [pixel]')
    ax.set_ylabel('y-axis [pixel]')
    ax.set_title(header['DATE-BEG'])
    my_mappabel = matplotlib.cm.ScalarMappable(cmap=my_colormap, norm=norm_SL)
    plt.colorbar(label='[MSB]')
    point_set = np.zeros([num_streaks, 2, 2], dtype=float)
    for i in range(num_streaks):
        [x_1, x_2] = plt.ginput(2, timeout=-1)
        ax.plot([x_1, x_2], color='red')
        point_set[i, 0, :] = np.array(x_1, dtype=int) + 1
        point_set[i, 1, :] = np.array(x_2, dtype=int) + 1  # +1 because the index of array in Python starts from 0
        ax.plot([point_set[i, 0, 0] - 1, point_set[i, 1, 0] - 1], [point_set[i, 0, 1] - 1, point_set[i, 1, 1] - 1], 'r')
        # Note that point_set[i, 0, 0]( x-axis ) represents the second axis of the FITS data(axis-1) while point_set[i,
        # 0, 1] represents the first axis of the FITS data(axis-0).
    plt.show()
    # Draw the WISPR_Inner map, and retrieve the raw data points of streaks by hands
    # Then, through these points get all lines in 2d frame and derive their interceptions
    k_and_b = np.ones([num_streaks, 2], dtype=float)
    for fun_i in range(num_streaks):
        k_and_b[fun_i, 0] = (point_set[fun_i, 1, 1] - point_set[fun_i, 0, 1]) / (
                    point_set[fun_i, 1, 0] - point_set[fun_i, 0, 0])
        # the second index '0' means the slope
        k_and_b[fun_i, 1] = point_set[fun_i, 1, 1] - k_and_b[fun_i, 0] * point_set[fun_i, 1, 0]
        # the second index '1' means the intercept
    return point_set, k_and_b


def get_vanishing_point(point_set, need_test=False):
    """
    :param point_set: n * 2 array, 2d coordinates of all the interceptions.
    :param need_test: if you want to examine the descent speed and check if it converges.
    :return: the center point of all point input. 1 * 2 array
    The algorithm in this function is k-means (using criterion of distance minimizing)
    """
    vanish_point, line_label, _ = cluster.k_means(point_set, n_clusters=1)
    vanish_point = np.array(vanish_point, dtype=float)
    if need_test is True:
        fun_fig = plt.figure()
        fun_ax = fun_fig.add_subplot(111)
        fun_ax.scatter(point_set, color='yellow')
        fun_ax.scatter(vanish_point[0], marker='x', color='black')
        plt.show()
    return vanish_point


def construct_3d_psp_tps_model(epoch):
    """
    :return: the TPS vertexes' coordinates in PSP HCI frame, origin located at WISPR-INNER.
    Since the whole 3d model is imported, this function is useless.     ——Tianhang Chen, 2022.3.18
    """
    fun_h = 1.91  # unit:m
    fun_thickness = 0.125
    fun_radius_cylinder = 0.5  # unit:m
    fun_radius_tps = 2.3  # unit:m
    # point set in the SPP_SPACECRAFT frame
    centroid = np.array([-0.84, -0.26, 0], dtype=float)
    edge_len = fun_radius_tps / np.sqrt(3)
    vertexes = np.ones([28, 3], dtype=float)
    vertexes[0] = np.array([1.16, 0, 1.91], dtype=float)
    vertexes[1] = np.array([1.145, 0.2, 1.91], dtype=float)
    vertexes[2] = np.array([0.95, 0.8, 1.91], dtype=float)
    vertexes[3] = np.array([0.625, 1.125, 1.91], dtype=float)
    vertexes[4] = np.array([-0.625, 1.125, 1.91], dtype=float)
    vertexes[5] = np.array([-0.95, 0.8, 1.91], dtype=float)
    vertexes[6] = np.array([-1.145, 0.2, 1.91], dtype=float)
    vertexes[7] = np.array([-1.16, 0, 1.91], dtype=float)
    vertexes[8] = np.array([-1.145, -0.2, 1.91], dtype=float)
    vertexes[9] = np.array([-0.95, -0.8, 1.91], dtype=float)
    vertexes[10] = np.array([-0.625, -1.125, 1.91], dtype=float)
    vertexes[11] = np.array([0.625, -1.125, 1.91], dtype=float)
    vertexes[12] = np.array([0.95, -0.8, 1.91], dtype=float)
    vertexes[13] = np.array([1.145, -0.2, 1.91], dtype=float)
    for fun_i in range(14):
        vertexes[fun_i] = vertexes[fun_i] + centroid
        vertexes[fun_i + 14] = vertexes[fun_i] - np.array([0, 0, fun_thickness], dtype=float)
    vertexes_HCI = np.ones([28, 3], dtype=float)
    for fun_i in range(28):
        temp_vertex, _ = spice.spkcpt(vertexes[fun_i], 'SPP', 'SPP_SPACECRAFT', epoch,
                                      'SPP_HCI', 'OBSERVER', 'NONE', 'SPP')
        vertexes_HCI[fun_i] = temp_vertex[0:3]
    return vertexes_HCI


def plot_FOV_and_FrameAxes(epoch_time):
    """
    :param epoch_time: the epoch(SPICE) of observing time.
    :return: fun_fov_traces and fun_axes_traces are both the 'trace' type introduced in plotly(https://plotly.com/python)
    """
    fun_fov_traces = []
    fun_axes_traces = []
    wispr_inner_parameter = spice.getfov(-96100, 4)
    for i_edge in range(4):
        edge_inner1, _ = spice.spkcpt(wispr_inner_parameter[4][i_edge], 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                      'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
        edge_motion = np.zeros([3, 1000], dtype=float)
        for cadence in range(1000):
            edge_motion[:, cadence] = (edge_inner1[0:3] * 2e1) * cadence / 6000
        edge_motion[0, :] = edge_motion[0, :] + WISPR_pos[0]
        edge_motion[1, :] = edge_motion[1, :] + WISPR_pos[1]
        edge_motion[2, :] = edge_motion[2, :] + WISPR_pos[2]
        fun_fov_traces.append(go.Scatter3d(
                                             x=edge_motion[0], y=edge_motion[1], z=edge_motion[2], mode='lines',
                                             opacity=0.6, line=dict(color='green', width=5),
                                             legendgroup='FOV',
                                             legendgrouptitle=dict(text='FOV of WISPR_Inner')
                                             )
                              )
    # ax_3.plot([0, the_center_point_3d[0]], [0, the_center_point_3d[1]], [0, the_center_point_3d[2]], c='orange'
    #           , label='vanishing point position')
    # the top(bottom) of streaks retrieved that converge at the y<0(y>0)
    # vanishing point in 3d frame.
    x_wispr_inner = np.array([2, 0, 0], dtype=float)
    x_sc, _ = spice.spkcpt(x_wispr_inner, 'SPP', 'SPP_WISPR_INNER', epoch_time, 'SPP_SPACECRAFT',
                           'OBSERVER', 'NONE', 'SPP')
    x_sc = np.array(x_sc, dtype=float)
    x_sc = x_sc[0:3] + WISPR_pos
    y_wispr_inner = np.array([0, 1, 0], dtype=float)
    y_sc, _ = spice.spkcpt(y_wispr_inner, 'SPP', 'SPP_WISPR_INNER', epoch_time, 'SPP_SPACECRAFT',
                           'OBSERVER', 'NONE', 'SPP')
    y_sc = np.array(y_sc, dtype=float)
    y_sc = y_sc[0:3] + WISPR_pos
    z_wispr_inner = np.array([0, 0, 4], dtype=float)
    z_sc, _ = spice.spkcpt(z_wispr_inner, 'SPP', 'SPP_WISPR_INNER', epoch_time, 'SPP_SPACECRAFT',
                           'OBSERVER', 'NONE', 'SPP')
    z_sc = np.array(z_sc, dtype=float)
    z_sc = z_sc[0:3] + WISPR_pos
    # axix_sc means the WISPR frame axis in spacecraft frame.
    fun_axes_traces.append(go.Scatter3d(
                                        x=np.array([WISPR_pos[0], x_sc[0]], dtype=float),
                                        y=np.array([WISPR_pos[1], x_sc[1]], dtype=float),
                                        z=np.array([WISPR_pos[2], x_sc[2]], dtype=float), mode='lines',
                                        opacity=0.6, line=dict(color='black', width=5), legendgroup='frame axis',
                                        legendgrouptitle=dict(text='WISPR_frame axis, longest '
                                                                   'for z, shortest for x')
                                        )
                           )
    fun_axes_traces.append(go.Scatter3d(
                                        x=np.array([WISPR_pos[0], y_sc[0]], dtype=float),
                                        y=np.array([WISPR_pos[1], y_sc[1]], dtype=float),
                                        z=np.array([WISPR_pos[2], y_sc[2]], dtype=float), mode='lines',
                                        opacity=0.6, line=dict(color='black', width=5), legendgroup='frame axis',
                                        legendgrouptitle=dict(text='WISPR_frame axis, longest '
                                                                   'for z, shortest for x')
                                        )
                           )
    fun_axes_traces.append(go.Scatter3d(
                                        x=np.array([WISPR_pos[0], z_sc[0]], dtype=float),
                                        y=np.array([WISPR_pos[1], z_sc[1]], dtype=float),
                                        z=np.array([WISPR_pos[2], z_sc[2]], dtype=float), mode='lines',
                                        opacity=0.6, line=dict(color='black', width=5), legendgroup='frame axis',
                                        legendgrouptitle=dict(text='WISPR_frame axis, longest '
                                                                   'for z, shortest for x')
                                        )
                           )
    return fun_fov_traces, fun_axes_traces


def plot_3d_trajectory(fun_center_point, line_orientation, epoch_time, the_flag):
    """
    the same as plot_FOV_adn_FrameAxes
    :param fun_center_point: ...
    :param line_orientation: ...
    :param epoch_time: ...
    :param the_flag
    :return: ...
    """
    trajectory_traces = []
    the_center_point_3d = np.array(
        [(fun_center_point[0, 0] - x_0_pixel) * alpha, (fun_center_point[0, 1] - y_0_pixel) * alpha,
         f], dtype=float) * 8e1
    the_center_point_3d, _ = spice.spkcpt(the_center_point_3d, 'SPP', 'SPP_WISPR_INNER', et,
                                          'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    the_center_point_3d = the_center_point_3d[0:3] + WISPR_pos
    if fun_center_point[0, 1] <= y_0_pixel / 2:  # if the centroid is located y<0
        extreme_point = all_line_points.min(axis=0)  # the min y represents the top of the fig
    else:
        extreme_point = all_line_points.max(axis=0)
    # extreme_point_3d_1, 2 represents the same point(end)in WISPR_inner, with a different scaling factor, respectively.
    # extreme_point_3d_3 represents the other point(start) in WISPR_inner, just for LOS plotting.
    extreme_point_3d_1 = np.array([
        (extreme_point[1, 0] - x_0_pixel) * alpha, (extreme_point[1, 1] - y_0_pixel) * alpha, f],
        dtype=float) * 8e1
    extreme_point_3d_1, _ = spice.spkcpt(extreme_point_3d_1, 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                         'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    extreme_point_3d_1 = extreme_point_3d_1[0:3] + WISPR_pos
    extreme_point_3d_2 = np.array([
        (extreme_point[1, 0] - x_0_pixel) * alpha, (extreme_point[1, 1] - y_0_pixel) * alpha, f],
        dtype=float) * 2
    extreme_point_3d_2, _ = spice.spkcpt(extreme_point_3d_2, 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                         'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    extreme_point_3d_2 = extreme_point_3d_2[0:3] + WISPR_pos
    extreme_point_3d_3 = np.array([
        (extreme_point[0, 0] - x_0_pixel) * alpha, (extreme_point[0, 1] - y_0_pixel) * alpha, f],
        dtype=float) * 8e1
    extreme_point_3d_3, _ = spice.spkcpt(extreme_point_3d_3, 'SPP', 'SPP_WISPR_INNER', epoch_time,
                                         'SPP_SPACECRAFT', 'OBSERVER', 'NONE', 'SPP')
    extreme_point_3d_3 = extreme_point_3d_3[0:3] + WISPR_pos
    trajectory_traces.append(
        go.Scatter3d(
                    x=np.array([extreme_point_3d_1[0], extreme_point_3d_1[0] - 4 * line_orientation[0]], dtype=float),
                    y=np.array([extreme_point_3d_1[1], extreme_point_3d_1[1] - 4 * line_orientation[1]], dtype=float),
                    z=np.array([extreme_point_3d_1[2], extreme_point_3d_1[2] - 4 * line_orientation[2]], dtype=float),
                    mode='lines', opacity=1, line=dict(color='blue', width=2), legendgroup='dust trace',
                    legendgrouptitle=dict(text='possible positions')
                    )
                            )
    trajectory_traces.append(
        go.Scatter3d(
            x=np.array([extreme_point_3d_2[0], extreme_point_3d_2[0] - 4 * line_orientation[0]], dtype=float),
            y=np.array([extreme_point_3d_2[1], extreme_point_3d_2[1] - 4 * line_orientation[1]], dtype=float),
            z=np.array([extreme_point_3d_2[2], extreme_point_3d_2[2] - 4 * line_orientation[2]], dtype=float),
            mode='lines', opacity=1, line=dict(color='blue', width=2), legendgroup='dust trace',
            legendgrouptitle=dict(text='possible positions')
        )
                            )
    trajectory_traces.append(
        go.Scatter3d(
            x=np.array([WISPR_pos[0], extreme_point_3d_1[0] * 2 - WISPR_pos[0]], dtype=float),
            y=np.array([WISPR_pos[1], extreme_point_3d_1[1] * 2 - WISPR_pos[1]], dtype=float),
            z=np.array([WISPR_pos[2], extreme_point_3d_1[2] * 2 - WISPR_pos[2]], dtype=float),
            mode='lines', opacity=1, line=dict(color='blue', dash='dash', width=2), legendgroup='LOS',
            legendgrouptitle=dict(text='Line of Sight')
        )
    )
    trajectory_traces.append(
        go.Scatter3d(
            x=np.array([WISPR_pos[0], extreme_point_3d_3[0] * 2 - WISPR_pos[0]], dtype=float),
            y=np.array([WISPR_pos[1], extreme_point_3d_3[1] * 2 - WISPR_pos[1]], dtype=float),
            z=np.array([WISPR_pos[2], extreme_point_3d_3[2] * 2 - WISPR_pos[2]], dtype=float),
            mode='lines', opacity=1, line=dict(color='blue', dash='dash', width=2), legendgroup='LOS',
            legendgrouptitle=dict(text='Line of Sight')
        )
    )
    if ~the_flag:
        trajectory_traces.append(
            go.Scatter3d(
                x=np.array([WISPR_pos[0], the_center_point_3d[0] * 2 - WISPR_pos[0]], dtype=float),
                y=np.array([WISPR_pos[1], the_center_point_3d[1] * 2 - WISPR_pos[1]], dtype=float),
                z=np.array([WISPR_pos[2], the_center_point_3d[2] * 2 - WISPR_pos[2]], dtype=float),
                mode='lines', opacity=1, line=dict(color='orange', width=3.5), legendgroup='centroid',
                legendgrouptitle=dict(text='Possible position of the cone origin')
            )
        )
    return trajectory_traces


data, header = sunpy.io.fits.read(fits_file_path)[0]
header['BUNIT'] = 'MSB'
a_map = sunpy.map.Map(data, header)
my_colormap = copy.deepcopy(a_map.cmap)
true_data = copy.deepcopy(data)
i = 0
fits_data_size_flag = False
# False means the FITS data is not compressed with a size of 2048*1920, while True with a size of 1024*960.
# It just depends on the FITS file.
if fits_data_size_flag:
    scaling_factor = 2
else:
    scaling_factor = 1

x_pixel_size = int(1920 / scaling_factor)
y_pixel_size = int(2048 / scaling_factor)

while i < y_pixel_size:
    j = 0
    while j < x_pixel_size:
        true_data[i, j] = data[y_pixel_size-1-i, j]
        j = j + 1
    i = i + 1

all_line_points, slope_and_intercept = retrieve_streaks(streak_number)
interceptions = []
i = count = 0
while i < streak_number:
    j = i + 1
    while j < streak_number:
        temp_interp_x = (slope_and_intercept[j, 1] - slope_and_intercept[i, 1]) / (slope_and_intercept[i, 0] -
                                                                                   slope_and_intercept[j, 0])
        temp_interp_y = slope_and_intercept[i, 0] * temp_interp_x + slope_and_intercept[i, 1]
        interceptions.append([temp_interp_x, temp_interp_y])
        j = j + 1
    i = i + 1
interceptions = np.array(interceptions, dtype=float)
the_center_point = get_vanishing_point(interceptions)


f = 28e-3  # the focal length of WISPR_INNER is 28mm
x_0_pixel = x_pixel_size / 2
y_0_pixel = y_pixel_size / 2  # midpoint
alpha = 1e-5 * scaling_factor
# the size of single CCD pixel is 0.01mm
# If the size of FITS data is half the true pixel size(2048*1920), the scaling factor 2 above is needed.

phi = np.arctan((the_center_point[0, 1] - y_0_pixel) / (the_center_point[0, 0] - x_0_pixel))
theta = np.arctan(np.cos(phi) / alpha / (the_center_point[0, 0] - x_0_pixel) * f)   # unit of phi and theta: rad
phi_deg = phi * 180 / np.pi
theta_deg = theta * 180 / np.pi
orientation_vec_WISPR_I = np.zeros([3, 1], dtype=float)
if line_flag:
    orientation_vec_WISPR_I[0] = (all_line_points[0, 1, 0] - all_line_points[0, 0, 0]) / \
                                 np.sqrt((all_line_points[0, 1, 0] - all_line_points[0, 0, 0]) ** 2 +
                                         (all_line_points[0, 1, 1] - all_line_points[0, 0, 1]) ** 2)
    orientation_vec_WISPR_I[1] = (all_line_points[0, 1, 1] - all_line_points[0, 0, 1]) / \
                                 np.sqrt((all_line_points[0, 1, 0] - all_line_points[0, 0, 0]) ** 2 +
                                         (all_line_points[0, 1, 1] - all_line_points[0, 0, 1]) ** 2)
    orientation_vec_WISPR_I[2] = 0
else:
    orientation_vec_WISPR_I[0] = np.cos(theta) * np.cos(phi)
    orientation_vec_WISPR_I[1] = np.cos(theta) * np.sin(phi)
    orientation_vec_WISPR_I[2] = np.sin(theta)
# The orientation of streaks in WISPR_INNER frame has been derived(orientation_vev_WISPR_I)
# The next step: transfer it from WISPR_I to SPP_SPACECRAFT frame
et, rotation_matrix = get_psp_state(header['DATE-BEG'], '%Y-%m-%dT%H:%M:%S.%f')
orientation_vec_sc = np.dot(rotation_matrix, orientation_vec_WISPR_I)

# Plot all.
transfer_factor = 2.25 / 0.219372 / 2
# the tranforming factor from the stl size to the true size of psp.
# 2.25 is the true size of TPS diameter(smallest), while 0.219372*2 is the stl size of it.    ——Tianhang Chen, 2022.3.18
plotly_trace_1 = add_psp_3d_model(np.zeros([3, 1], dtype=float), rot_theta=0, scale=transfer_factor)
plotly_trace_2, plotly_trace_3 = plot_FOV_and_FrameAxes(et)
plotly_trace_4 = plot_3d_trajectory(the_center_point, orientation_vec_sc, et, line_flag)
plotly_fig = go.Figure()
plotly_fig.add_trace(plotly_trace_1)
for i in range(len(plotly_trace_2)):
    plotly_fig.add_trace(plotly_trace_2[i])
for i in range(len(plotly_trace_4)):
    plotly_fig.add_trace(plotly_trace_4[i])
for i in range(len(plotly_trace_3)):
    plotly_fig.add_trace(plotly_trace_3[i])
plotly_fig.update_layout(title_text="Spacecraft Frame" + ' ' + header['DATE-BEG'],
                         title_font_size=30)
plotly_fig.show()

