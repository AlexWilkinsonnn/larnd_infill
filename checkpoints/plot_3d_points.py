import argparse
from itertools import cycle

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

    norm_feats = matplotlib.colors.Normalize(
        vmin=min(feat[0] for feat in target_points.values()),
        vmax=max(feat[0] for feat in target_points.values())
    )
    m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.viridis)

    x_size = vmap["x_step_target"]
    y_size = vmap["y_step_target"]
    z_size = vmap["z_step_target"]
    x_size_draw = x_size / 5
    y_size_draw = x_size / 5
    z_size_draw = z_size / 2
    for coords, feat in pred_points.items():
        feat = int(feat[0])
        if feat < 16:
            continue

        x_bin = vmap["x"][coords[0]]
        x_pos = (x_bin[1] + x_bin[0]) / 2
        y_bin = vmap["y"][coords[1]]
        y_pos = (y_bin[1] + y_bin[0]) / 2
        z_bin = vmap["z"][coords[2]]
        z_pos = (z_bin[1] + z_bin[0]) / 2

        x, y, z = get_cube()
        x = x * x_size_draw + (x_pos * x_size)
        y = y * y_size_draw + (y_pos * y_size)
        z = z * z_size_draw + (z_pos * z_size)

        c = m_feats.to_rgba(feat)

        ax.plot_surface(x, z, y, color=c, shade=False)

    ax.set_box_aspect((4,8,4))
    ax.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

