import numpy as np
import matplotlib; from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d


def plot_ndlar(packets, detector, tracks=None):
    """Plots ND-LAr with packets from a single event and optionally tracks"""
    xy_size = detector.pixel_pitch
    z_size = detector.time_sampling * detector.vdrift

    norm_adc = matplotlib.colors.Normalize(vmin=0, vmax=300)
    m_adc = matplotlib.cm.ScalarMappable(norm=norm_adc, cmap=matplotlib.cm.jet)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for p in packets:
        # if p.z() > 50.4:
        #     continue
        x, y, z = get_cube()
        x = x * xy_size + p.x + p.anode.tpc_x
        y = y * xy_size + p.y + p.anode.tpc_y
        z = z * z_size + p.z_global()
        ax.plot_surface(x, z, y, color=m_adc.to_rgba(p.ADC))

    if tracks is not None:
        for t in tracks:
            ax.plot(
                (t.x_start, t.x_end), (t.y_start, t.y_end), (t.z_start, t.z_end),
                color="r", linewidth=0.5
            )

    # Magic code to draw ND-LAr modules taken from larnd-sim example
    for i in range(0, 70, 2):
        anode1 = plt.Rectangle(
            (detector.tpc_borders[i][0][0], detector.tpc_borders[i][1][0]),
            detector.tpc_borders[i][0][1] - detector.tpc_borders[i][0][0],
            detector.tpc_borders[i][1][1] - detector.tpc_borders[i][1][0],
            linewidth=1, fc='none', edgecolor='gray'
        )
        ax.add_patch(anode1)
        art3d.pathpatch_2d_to_3d(anode1, z=detector.tpc_borders[0][2][0], zdir="y")

        anode2 = plt.Rectangle(
            (detector.tpc_borders[i][0][0], detector.tpc_borders[i][1][0]),
            detector.tpc_borders[i][0][1] - detector.tpc_borders[i][0][0],
            detector.tpc_borders[i][1][1] - detector.tpc_borders[i][1][0],
            linewidth=1, fc='none', edgecolor='gray'
        )
        ax.add_patch(anode2)
        art3d.pathpatch_2d_to_3d(anode2, z=detector.tpc_borders[i+1][2][0], zdir="y")

        cathode = plt.Rectangle(
            (detector.tpc_borders[i][0][0], detector.tpc_borders[i][1][0]),
            detector.tpc_borders[i][0][1] - detector.tpc_borders[i][0][0],
            detector.tpc_borders[i][1][1] - detector.tpc_borders[i][1][0],
            linewidth=1, fc='gray', alpha=0.2, edgecolor='gray'
        )
        ax.add_patch(cathode)
        z_cathode = (detector.tpc_borders[i][2][0]+detector.tpc_borders[i+1][2][0])/2
        art3d.pathpatch_2d_to_3d(cathode, z=z_cathode, zdir="y")

        ax.plot(
            (detector.tpc_borders[i][0][0], detector.tpc_borders[i][0][0]),
            (detector.tpc_borders[i][2][0], detector.tpc_borders[i+1][2][0]),
            (detector.tpc_borders[i][1][0], detector.tpc_borders[i][1][0]),
            lw=1, color='gray'
        )

        ax.plot(
            (detector.tpc_borders[i][0][0], detector.tpc_borders[i][0][0]),
            (detector.tpc_borders[i][2][0], detector.tpc_borders[i+1][2][0]),
            (detector.tpc_borders[i][1][1], detector.tpc_borders[i][1][1]),
            lw=1, color='gray'
        )

        ax.plot(
            (detector.tpc_borders[i][0][1], detector.tpc_borders[i][0][1]),
            (detector.tpc_borders[i][2][0], detector.tpc_borders[i+1][2][0]),
            (detector.tpc_borders[i][1][0], detector.tpc_borders[i][1][0]),
            lw=1, color='gray'
        )

        ax.plot(
            (detector.tpc_borders[i][0][1], detector.tpc_borders[i][0][1]),
            (detector.tpc_borders[i][2][0], detector.tpc_borders[i+1][2][0]),
            (detector.tpc_borders[i][1][1], detector.tpc_borders[i][1][1]),
            lw=1, color='gray'
        )

    ax.set_xlim(detector.tpc_borders[0][0][0],detector.tpc_borders[-1][0][1])
    ax.set_ylim(detector.tpc_borders[0][2][0],detector.tpc_borders[-1][2][0])
    ax.set_zlim(detector.tpc_borders[0][1][0],detector.tpc_borders[-1][1][1])
    ax.set_box_aspect((4,8,4))
    ax.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.show()


def get_cube():
    """Get coords for plotting cuboid surface with Axes3D.plot_surface"""
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)

    return x,y,z


def plot_ndlar_voxels(
    coords, adcs, detector,
    pix_cols_per_anode=256, pix_cols_per_gap=11,
    pix_rows_per_anode=800,
    ticks_per_module=6116, ticks_per_gap=79
):
    """
    Plot ND-LAr from data that has been voxelised
    (array is too large to use Axes3D.voxels so drawing surfaces)
    """
    xy_size = detector.pixel_pitch
    z_size = detector.time_sampling * detector.vdrift

    norm_adc = matplotlib.colors.Normalize(vmin=0, vmax=300)
    m_adc = matplotlib.cm.ScalarMappable(norm=norm_adc, cmap=matplotlib.cm.jet)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for coord_x, coord_y, coord_z, adc in zip(*coords, adcs):
        x, y, z = get_cube()
        x = x * xy_size + (coord_x * xy_size)
        y = y * xy_size + (coord_y * xy_size)
        z = z * z_size + (coord_z * z_size)

        ax.plot_surface(x, z, y, color=m_adc.to_rgba(adc))

    x_max = (pix_cols_per_anode * 5 + (pix_cols_per_gap * 4)) * xy_size
    y_max = pix_rows_per_anode * xy_size
    z_max = (ticks_per_module * 7 + (ticks_per_gap * 6)) * z_size
    for i in range(4):
        for coord_x in [
            pix_cols_per_anode * (i + 1), pix_cols_per_anode * (i + 1) + pix_cols_per_gap
        ]:
            x = coord_x * xy_size
            ax.plot((x, x), (0, 0), (0, y_max), color="black", lw=0.5)
            ax.plot((x, x), (0, z_max), (y_max, y_max), color="black", lw=0.5)
            ax.plot((x, x), (z_max, z_max), (y_max, 0), color="black", lw=0.5)
            ax.plot((x, x), (z_max, 0), (0, 0), color="black", lw=0.5)

    for i in range(6):
        for coord_z in [ticks_per_module * (i + 1), ticks_per_module * (i + 1) + ticks_per_gap]:
            z = coord_z * z_size
            ax.plot((0, 0), (z, z), (0, y_max), color="black", lw=0.5)
            ax.plot((0, x_max), (z, z), (y_max, y_max),color="black", lw=0.5)
            ax.plot((x_max, x_max), (z, z), (y_max, 0), color="black", lw=0.5)
            ax.plot((x_max, 0), (z, z),(0, 0),  color="black", lw=0.5)

    ax.set_xlim(0, detector.tpc_borders[-1][0][1] - detector.tpc_borders[0][0][0])
    ax.set_ylim(0, detector.tpc_borders[-1][2][0] - detector.tpc_borders[0][2][0])
    ax.set_zlim(0, detector.tpc_borders[-1][1][1] - detector.tpc_borders[0][1][0])
    ax.set_box_aspect((4,8,4))
    ax.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.show()

