import numpy as np
import matplotlib; from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

def plot_ndlar(packets, detector, tracks=None, structures=True, projections=False):
    """Plots ND-LAr with packets from a single event and optionally tracks"""
    xy_size = detector.pixel_pitch
    z_size = detector.time_sampling * detector.vdrift

    norm_adc = matplotlib.colors.Normalize(vmin=0, vmax=300)
    m_adc = matplotlib.cm.ScalarMappable(norm=norm_adc, cmap=matplotlib.cm.jet)

    if projections:
        fig, ax = plt.subplots(1, 3)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    max_z_local = max(p.z() for p in packets)
    z_local_shift = min(50.4 - max_z_local, 0)
    print(z_local_shift)
    print(sorted(p.z() for p in packets)[-10:])
    if projections:
        for p in packets:
            if p.z() > 60:
                print(p.z(), p.ADC, p.x, p.y)
                print(p.t_0, p.timestamp)
                continue
            # Cant remember if these are centres of pixels of corners?
            x = p.x + p.anode.tpc_x - (xy_size / 2)
            y = p.y + p.anode.tpc_y - (xy_size / 2)
            z = p.z_global() + z_local_shift
            adc = p.ADC
            # print(p.x, p.y, p.z())
            z_size_scaled = z_size * 5
            ax[0].add_patch(
                matplotlib.patches.Rectangle((x, y), xy_size, xy_size, fc=m_adc.to_rgba(adc))
            )
            ax[1].add_patch(
                matplotlib.patches.Rectangle((x, z), xy_size, z_size_scaled, fc=m_adc.to_rgba(adc))
            )
            ax[2].add_patch(
                matplotlib.patches.Rectangle((z, y), z_size_scaled, xy_size, fc=m_adc.to_rgba(adc))
            )
        if tracks is not None:
            for t in tracks:
                ax[0].plot((t.x_start, t.x_end), (t.y_start, t.y_end), color="r", linewidth=0.5)
                ax[1].plot((t.x_start, t.x_end), (t.z_start, t.z_end), color="r", linewidth=0.5)
                ax[2].plot((t.z_start, t.z_end), (t.y_start, t.y_end), color="r", linewidth=0.5)

    else:
        for p in packets:
            if p.z() > 50.4:
                continue
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
    if structures:
        if projections:
            raise NotImplementedError
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

    if projections:
        ax[0].set_xlim(detector.tpc_borders[-1][0][1], detector.tpc_borders[0][0][0])
        ax[1].set_xlim(detector.tpc_borders[-1][0][1], detector.tpc_borders[0][0][0])
        ax[2].set_xlim(detector.tpc_borders[-1][2][0], detector.tpc_borders[0][2][0])
        ax[0].set_ylim(detector.tpc_borders[-1][1][1], detector.tpc_borders[0][1][0])
        ax[1].set_ylim(detector.tpc_borders[-1][2][0], detector.tpc_borders[0][2][0])
        ax[2].set_ylim(detector.tpc_borders[-1][1][1], detector.tpc_borders[0][1][0])
        ax[0].set_xlabel("x")
        ax[1].set_xlabel("x")
        ax[2].set_xlabel("z")
        ax[0].set_ylabel("y")
        ax[1].set_ylabel("z")
        ax[2].set_ylabel("y")
    else:
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
    ticks_per_module=6117, ticks_per_gap=79,
    infill_coords=None,
    structure=True,
    projections=False,
    z_downsample=1
):
    """
    Plot ND-LAr from data that has been voxelised
    (array is too large to use Axes3D.voxels so drawing surfaces)
    """
    xy_size = detector.pixel_pitch
    z_size = detector.time_sampling * detector.vdrift * z_downsample
    z_size_scale = 5 if z_downsample == 1 else 1

    norm_adc = matplotlib.colors.Normalize(vmin=0, vmax=300)
    m_adc = matplotlib.cm.ScalarMappable(norm=norm_adc, cmap=matplotlib.cm.jet)

    if projections:
        fig, ax = plt.subplots(1, 3)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    if infill_coords is not None:
        for coord_x, coord_y, coord_z in zip(*infill_coords):
            if projections:
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(
                        (coord_x * xy_size, coord_y * xy_size), xy_size, xy_size, fc="green"
                    )
                )
                ax[1].add_patch(
                    matplotlib.patches.Rectangle(
                        (coord_x * xy_size, coord_z * z_size), xy_size, z_size, fc="green"
                    )
                )
                ax[2].add_patch(
                    matplotlib.patches.Rectangle(
                        (coord_z * z_size, coord_y * xy_size), z_size, xy_size, fc="green"
                    )
                )

            else:
                x = x * xy_size + (coord_x * xy_size)
                y = y * xy_size + (coord_y * xy_size)
                z = z * z_size + (coord_z * z_size)
                ax.plot_surface(x, z, y, color="green")

    for coord_x, coord_y, coord_z, adc in zip(*coords, adcs):
        if projections:
            ax[0].add_patch(
                matplotlib.patches.Rectangle(
                    (coord_x * xy_size, coord_y * xy_size), xy_size, xy_size, fc=m_adc.to_rgba(adc)                )
            )
            ax[1].add_patch(
                matplotlib.patches.Rectangle(
                    (coord_x * xy_size, coord_z * z_size),
                    xy_size, z_size * z_size_scale, fc=m_adc.to_rgba(adc)
                )
            )
            ax[2].add_patch(
                matplotlib.patches.Rectangle(
                    (coord_z * z_size, coord_y * xy_size),
                    z_size * z_size_scale, xy_size, fc=m_adc.to_rgba(adc)
                )
            )

        else:
            x, y, z = get_cube()
            x = x * xy_size + (coord_x * xy_size)
            y = y * xy_size + (coord_y * xy_size)
            z = z * z_size + (coord_z * z_size)
            ax.plot_surface(x, z, y, color=m_adc.to_rgba(adc))

    if structure:
        if projections:
            raise NotImplementedError
        if z_downsample != 1:
            raise NotImplementedError
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
            for coord_z in [
                ticks_per_module * (i + 1), ticks_per_module * (i + 1) + ticks_per_gap
            ]:
                z = coord_z * z_size
                ax.plot((0, 0), (z, z), (0, y_max), color="black", lw=0.5)
                ax.plot((0, x_max), (z, z), (y_max, y_max),color="black", lw=0.5)
                ax.plot((x_max, x_max), (z, z), (y_max, 0), color="black", lw=0.5)
                ax.plot((x_max, 0), (z, z),(0, 0),  color="black", lw=0.5)

    if projections:
        ax[0].set_xlim(0, detector.tpc_borders[-1][0][1] - detector.tpc_borders[0][0][0])
        ax[1].set_xlim(0, detector.tpc_borders[-1][0][1] - detector.tpc_borders[0][0][0])
        ax[2].set_xlim(0, detector.tpc_borders[-1][2][0] - detector.tpc_borders[0][2][0])
        ax[0].set_ylim(0, detector.tpc_borders[-1][1][1] - detector.tpc_borders[0][1][0])
        ax[1].set_ylim(0, detector.tpc_borders[-1][2][0] - detector.tpc_borders[0][2][0])
        ax[2].set_ylim(0, detector.tpc_borders[-1][1][1] - detector.tpc_borders[0][1][0])
        ax[0].set_xlabel("x")
        ax[1].set_xlabel("x")
        ax[2].set_xlabel("z")
        ax[0].set_ylabel("y")
        ax[1].set_ylabel("z")
        ax[2].set_ylabel("y")
    else:
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

def plot_ndlar_voxels_2(
    coords, feats, detector, x_vmap, y_vmap, z_vmap, x_gaps, z_gaps,
    z_scalefactor=1, max_feat=300, min_feat=0, saveas=None,
    tracks=None, signal_mask_active_coords=None, signal_mask_gap_coords=None,
    target_coords=None, target_feats=None,
    single_proj_pretty=False,
    plot_3d=False,
    autocrop=False
):
    norm_feats = matplotlib.colors.Normalize(vmin=min_feat, vmax=max_feat)
    if target_coords is None:
        m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.jet)
    else:
        m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.Blues)
        m_feats_target = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.Reds)

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    for x_gap_coord in x_gaps:
        x_bin = x_vmap[x_gap_coord]
        x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
        y_pos = detector.tpc_borders[-1][1][1]
        y_size = detector.tpc_borders[0][1][0] - detector.tpc_borders[-1][1][1]
        z_pos = detector.tpc_borders[-1][2][0]
        z_size = detector.tpc_borders[0][2][0] - detector.tpc_borders[-1][2][0]
        ax[0].add_patch(
            matplotlib.patches.Rectangle((x_pos, y_pos), x_size, y_size, fc="gray", alpha=0.3)
        )
        ax[1].add_patch(
            matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="gray", alpha=0.3)
        )

    for z_gap_coord in z_gaps:
        z_bin = z_vmap[z_gap_coord]
        z_size, z_pos = z_bin[1] - z_bin[0], z_bin[0]
        x_pos = detector.tpc_borders[-1][0][1]
        x_size = detector.tpc_borders[0][0][0] - detector.tpc_borders[-1][0][1]
        y_pos = detector.tpc_borders[-1][1][1]
        y_size = detector.tpc_borders[0][1][0] - detector.tpc_borders[-1][1][1]
        ax[1].add_patch(
            matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="gray", alpha=0.3)
        )
        ax[2].add_patch(
            matplotlib.patches.Rectangle((z_pos, y_pos), z_size, y_size, fc="gray", alpha=0.3)
        )

    curr_patches_xy, curr_patches_xz, curr_patches_zy = set(), set(), set()
    for coord_x, coord_y, coord_z, feat in zip(*coords, feats):
        x_bin = x_vmap[coord_x]
        x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
        y_bin = y_vmap[coord_y]
        y_size, y_pos = (y_bin[1] - y_bin[0]), y_bin[0]
        z_bin = z_vmap[coord_z]
        z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

        c = m_feats.to_rgba(feat)
        alpha = 1.0

        pos_xy = (x_pos, y_pos)
        if pos_xy not in curr_patches_xy:
            curr_patches_xy.add(pos_xy)
            ax[0].add_patch(
                matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha)
            )
        pos_xz = (x_pos, z_pos)
        if pos_xz not in curr_patches_xz:
            curr_patches_xz.add(pos_xz)
            ax[1].add_patch(
                matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
            )
        pos_zy = (z_pos, y_pos)
        if pos_zy not in curr_patches_zy:
            curr_patches_zy.add(pos_zy)
            ax[2].add_patch(
                matplotlib.patches.Rectangle(pos_zy, z_size, y_size, fc=c, alpha=alpha)
            )

    if target_coords is not None:
        for coord_x, coord_y, coord_z, feat in zip(*target_coords, target_feats):
            x_bin = x_vmap[coord_x]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            y_bin = y_vmap[coord_y]
            y_size, y_pos = (y_bin[1] - y_bin[0]) * 2, y_bin[0]
            z_bin = z_vmap[coord_z]
            z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

            c = m_feats_target.to_rgba(feat)
            alpha = 1.0

            pos_xy = (x_pos, y_pos)
            if pos_xy not in curr_patches_xy:
                curr_patches_xy.add(pos_xy)
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha)
                )
            pos_xz = (x_pos, z_pos)
            if pos_xz not in curr_patches_xz:
                curr_patches_xz.add(pos_xz)
                ax[1].add_patch(
                    matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
                )
            pos_zy = (z_pos, y_pos)
            if pos_zy not in curr_patches_zy:
                curr_patches_zy.add(pos_zy)
                ax[2].add_patch(
                    matplotlib.patches.Rectangle(pos_zy, z_size, y_size, fc=c, alpha=alpha)
                )

    # Now draw the signal mask patches
    curr_patches_mask_gap_xy, curr_patches_mask_gap_xz, curr_patches_mask_gap_zy = set(), set(), set()
    if signal_mask_gap_coords is not None:
        for coord_x, coord_y, coord_z in zip(*signal_mask_gap_coords):
            x_bin = x_vmap[coord_x]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            y_bin = y_vmap[coord_y]
            y_size, y_pos = y_bin[1] - y_bin[0], y_bin[0]
            z_bin = z_vmap[coord_z]
            z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

            c = "red"
            alpha = 0.3

            pos_xy = (x_pos, y_pos)
            if pos_xy not in curr_patches_xy and pos_xy not in curr_patches_mask_gap_xy:
                curr_patches_mask_gap_xy.add(pos_xy)
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha)
                )
            pos_xz = (x_pos, z_pos)
            if pos_xz not in curr_patches_xz and pos_xz not in curr_patches_mask_gap_xz:
                curr_patches_mask_gap_xz.add(pos_xz)
                ax[1].add_patch(
                    matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
                )
            pos_zy = (z_pos, y_pos)
            if pos_zy not in curr_patches_zy and pos_zy not in curr_patches_mask_gap_zy:
                curr_patches_mask_gap_zy.add(pos_zy)
                ax[2].add_patch(
                    matplotlib.patches.Rectangle(pos_zy, z_size, y_size, fc=c, alpha=alpha)
                )

    curr_patches_mask_active_xy, curr_patches_mask_active_xz, curr_patches_mask_active_zy = set(), set(), set()
    if signal_mask_active_coords is not None:
        for coord_x, coord_y, coord_z in zip(*signal_mask_active_coords):
            x_bin = x_vmap[coord_x]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            y_bin = y_vmap[coord_y]
            y_size, y_pos = y_bin[1] - y_bin[0], y_bin[0]
            z_bin = z_vmap[coord_z]
            z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

            c = "green"
            alpha = 0.3

            pos_xy = (x_pos, y_pos)
            if pos_xy not in curr_patches_xy and pos_xy not in curr_patches_mask_active_xy:
                if pos_xy in curr_patches_mask_gap_xy:
                    c = "orange"
                curr_patches_mask_active_xy.add(pos_xy)
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha)
                )
            pos_xz = (x_pos, z_pos)
            if pos_xz not in curr_patches_xz and pos_xz not in curr_patches_mask_active_xz:
                if pos_xz in curr_patches_mask_gap_xz:
                    c = "orange"
                curr_patches_mask_active_xz.add(pos_xz)
                ax[1].add_patch(
                    matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
                )
            pos_zy = (z_pos, y_pos)
            if pos_zy not in curr_patches_zy and pos_zy not in curr_patches_mask_active_zy:
                if pos_zy in curr_patches_mask_gap_zy:
                    c = "orange"
                curr_patches_mask_active_zy.add(pos_zy)
                ax[2].add_patch(
                    matplotlib.patches.Rectangle(pos_zy, z_size, y_size, fc=c, alpha=alpha)
                )


    if tracks is not None:
        for t in tracks:
            ax[0].plot((t.x_start, t.x_end), (t.y_start, t.y_end), color="r", linewidth=0.5)
            ax[1].plot((t.x_start, t.x_end), (t.z_start, t.z_end), color="r", linewidth=0.5)
            ax[2].plot((t.z_start, t.z_end), (t.y_start, t.y_end), color="r", linewidth=0.5)

    if autocrop:
        max_x = min(x_vmap[max(coords[0])][0] + 20, detector.tpc_borders[-1][0][1])
        min_x = max(x_vmap[min(coords[0])][0] - 20, detector.tpc_borders[0][0][0])
        max_y = min(y_vmap[max(coords[1])][0] + 20, detector.tpc_borders[-1][1][1])
        min_y = max(y_vmap[min(coords[1])][0] - 20, detector.tpc_borders[0][1][0])
        max_z = min(z_vmap[max(coords[2])][0] + 20, detector.tpc_borders[-1][2][0])
        min_z = max(z_vmap[min(coords[2])][0] - 20, detector.tpc_borders[0][2][0])
    else:
        max_x = detector.tpc_borders[0][0][0]
        min_x = detector.tpc_borders[-1][0][1]
        max_y = detector.tpc_borders[0][1][0]
        min_y = detector.tpc_borders[-1][1][1]
        max_z = detector.tpc_borders[0][2][0]
        min_z = detector.tpc_borders[-1][2][0]

    ax[0].set_xlim(min_x, max_x)
    ax[1].set_xlim(min_x, max_x)
    ax[2].set_xlim(min_z, max_z)
    ax[0].set_ylim(min_y, max_y)
    ax[1].set_ylim(min_z, max_z)
    ax[2].set_ylim(min_y, max_y)
    ax[0].set_xlabel("x")
    ax[1].set_xlabel("x")
    ax[2].set_xlabel("z")
    ax[0].set_ylabel("y")
    ax[1].set_ylabel("z")
    ax[2].set_ylabel("y")

    if single_proj_pretty:
        ax[0].set_xlim(540, 720)
        ax[0].set_ylim(-150, -10)
        ax[0].set_xlabel("x", fontsize=18)
        ax[0].set_ylabel("y", fontsize=18)

    # fig.tight_layout()
    if saveas is not None:
        plt.savefig(saveas, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for coord_x, coord_y, coord_z, feat in zip(*coords, feats):
            x_bin = x_vmap[coord_x]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            y_bin = y_vmap[coord_y]
            y_size, y_pos = (y_bin[1] - y_bin[0]), y_bin[0]
            z_bin = z_vmap[coord_z]
            z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

            x, y, z = get_cube()
            x = x * x_size + (x_pos * x_size)
            y = y * y_size + (y_pos * y_size)
            z = z * z_size + (z_pos * z_size)

            c = m_feats.to_rgba(feat)
            alpha = 1.0

            ax.plot_surface(x, z, y, color=c)

        # ax.set_xlim(detector.tpc_borders[-1][0][1], detector.tpc_borders[0][0][0])
        # ax.set_ylim(detector.tpc_borders[-1][2][0], detector.tpc_borders[0][2][0])
        # ax.set_zlim(detector.tpc_borders[-1][1][1], detector.tpc_borders[0][1][0])
        ax.set_xlim(212, 225)
        ax.set_ylim(12, 18)
        ax.set_zlim(-30, -45)
        ax.set_box_aspect((4,8,4))
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        plt.show()

