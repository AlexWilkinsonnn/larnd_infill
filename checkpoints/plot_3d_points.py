import argparse

import matplotlib; from matplotlib import pyplot as plt
import yaml
import numpy as np


def main(args):
    with open(args.voxel_map, "r") as f:
        vmap = yaml.load(f, yaml.FullLoader)
    with open(args.target_points_yml, "r") as f:
        target_points = yaml.load(f, yaml.FullLoader)
    with open(args.in_points_yml, "r") as f:
        in_points = yaml.load(f, yaml.FullLoader)
    with open(args.pred_points_yml, "r") as f:
        pred_points = yaml.load(f, yaml.FullLoader)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # norm_feats = matplotlib.colors.Normalize(
    #     vmin=-max(feat[0] for feat in target_points.values()),
    #     vmax=max(feat[0] for feat in target_points.values())
    # )
    # m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.coolwarm)

    x_size = vmap["x_step_target"]
    y_size = vmap["y_step_target"]
    z_size = vmap["z_step_target"]
    x_size_draw = x_size
    y_size_draw = y_size
    z_size_draw = z_size
    gap_coords = set()
    for coords, feat in in_points.items():
        adc = int(float(feat[0]) * (1 / args.in_scalefactor))

        x_bin = vmap["x"][coords[0]]
        x_pos = (x_bin[1] + x_bin[0]) / 2
        y_bin = vmap["y"][coords[1]]
        y_pos = (y_bin[1] + y_bin[0]) / 2
        z_bin = vmap["z"][coords[2]]
        z_pos = (z_bin[1] + z_bin[0]) / 2

        if int(feat[-1]):
            gap_coords.add((x_bin, y_bin, z_bin))

        if adc < 6:
            continue

        if (
            not (args.xlow < x_pos < args.xhigh) or
            not (args.ylow < y_pos < args.yhigh) or
            not (args.zlow < z_pos < args.zhigh)
        ):
            continue

        x, y, z = get_cube()
        x = x * x_size_draw + x_pos
        y = y * y_size_draw + y_pos
        z = z * z_size_draw + z_pos

        # c = m_feats.to_rgba(adc)

        ax.plot_surface(x, z, y, color="Blue", shade=False)

    for coords, feat in pred_points.items():
        adc = int(feat[0])
        if adc < 6:
            continue

        x_bin = vmap["x"][coords[0]]
        x_pos = (x_bin[1] + x_bin[0]) / 2
        y_bin = vmap["y"][coords[1]]
        y_pos = (y_bin[1] + y_bin[0]) / 2
        z_bin = vmap["z"][coords[2]]
        z_pos = (z_bin[1] + z_bin[0]) / 2

        if (x_bin, y_bin, z_bin) not in gap_coords:
            continue

        if (
            not (args.xlow < x_pos < args.xhigh) or
            not (args.ylow < y_pos < args.yhigh) or
            not (args.zlow < z_pos < args.zhigh)
        ):
            continue

        x, y, z = get_cube()
        x = x * x_size_draw + x_pos
        y = y * y_size_draw + y_pos
        z = z * z_size_draw + z_pos

        # c = m_feats.to_rgba(-adc)

        ax.plot_surface(x, z, y, color="Green", shade=False)

    ax.set_box_aspect((4,8,4))
    ax.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    plt.show()

def get_cube():
    """Get coords for plotting cuboid surface with Axes3D.plot_surface"""
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)

    return x,y,z

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("pred_points_yml", type=str)
    parser.add_argument("target_points_yml", type=str)
    parser.add_argument("in_points_yml", type=str)
    parser.add_argument("voxel_map", type=str)

    parser.add_argument("--xlow", type=float, default=-np.inf)
    parser.add_argument("--xhigh", type=float, default=np.inf)
    parser.add_argument("--ylow", type=float, default=-np.inf)
    parser.add_argument("--yhigh", type=float, default=np.inf)
    parser.add_argument("--zlow", type=float, default=-np.inf)
    parser.add_argument("--zhigh", type=float, default=np.inf)
    parser.add_argument("--in_scalefactor", type=float, default=1.0)

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
