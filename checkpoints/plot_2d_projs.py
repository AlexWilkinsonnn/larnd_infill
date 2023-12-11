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

    norm_feats = matplotlib.colors.Normalize(
        vmin=min(feat[0] for feat in target_points.values()),
        vmax=max(feat[0] for feat in target_points.values())
    )
    m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.viridis)

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    signal_mask_patches = set()
    for coords, feat in in_points.items():
        if args.proj == "xy":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["y"][coords[1]]
        elif args.proj == "xz":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["z"][coords[2]]
        elif args.proj == "zy":
            x_bin = vmap["z"][coords[2]]
            y_bin = vmap["y"][coords[1]]

        x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
        y_size, y_pos = y_bin[1] - y_bin[0], y_bin[0]

        pos_xy = (x_pos, y_pos)
        if feat[-1] and feat[0] == 0:
            c = "green"
            alpha = 0.3
            if pos_xy in signal_mask_patches:
                continue
            else:
                signal_mask_patches.add(pos_xy)
        elif feat[0] != 0:
            c = m_feats.to_rgba(feat[0] / args.adc_scalefactor)
            alpha = 1.0
        else:
            continue

        ax.add_patch(matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha))

    min_in_nonzero = min(feat[0] for feat in in_points.values() if feat[0] != 0)

    for coords, feat in pred_points.items():
        if args.proj == "xy":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["y"][coords[1]]
        elif args.proj == "xz":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["z"][coords[2]]
        elif args.proj == "zy":
            x_bin = vmap["z"][coords[2]]
            y_bin = vmap["y"][coords[1]]

        x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
        y_size, y_pos = y_bin[1] - y_bin[0], y_bin[0]

        if feat[0] < min_in_nonzero:
            continue
        c = m_feats.to_rgba(feat[0])
        alpha = 1.0

        pos_xy = (x_pos, y_pos)
        if pos_xy not in signal_mask_patches:
            continue;

        ax.add_patch(matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha))

    if args.proj == "xy":
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
    if args.proj == "xz":
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("z", fontsize=18)
    if args.proj == "zy":
        ax.set_xlabel("z", fontsize=18)
        ax.set_ylabel("y", fontsize=18)

    # fig.tight_layout()
    ax.set_xlim(500, 900)
    ax.set_ylim(-300, 300)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))

    signal_mask_patches = set()
    for coords, feat in in_points.items():
        if args.proj == "xy":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["y"][coords[1]]
        elif args.proj == "xz":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["z"][coords[2]]
        elif args.proj == "zy":
            x_bin = vmap["z"][coords[2]]
            y_bin = vmap["y"][coords[1]]

        x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
        y_size, y_pos = y_bin[1] - y_bin[0], y_bin[0]

        pos_xy = (x_pos, y_pos)
        if feat[-1] and feat[0] == 0:
            c = "green"
            alpha = 0.3
            if pos_xy in signal_mask_patches:
                continue
            else:
                signal_mask_patches.add(pos_xy)
        elif feat[0] != 0:
            c = m_feats.to_rgba(feat[0] / args.adc_scalefactor)
            alpha = 1.0
        else:
            continue

        ax.add_patch(matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha))

    for coords, feat in target_points.items():
        if args.proj == "xy":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["y"][coords[1]]
        elif args.proj == "xz":
            x_bin = vmap["x"][coords[0]]
            y_bin = vmap["z"][coords[2]]
        elif args.proj == "zy":
            x_bin = vmap["z"][coords[2]]
            y_bin = vmap["y"][coords[1]]

        x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
        y_size, y_pos = y_bin[1] - y_bin[0], y_bin[0]

        if feat[0] < min_in_nonzero:
            continue
        c = m_feats.to_rgba(feat[0])
        alpha = 1.0

        pos_xy = (x_pos, y_pos)
        if pos_xy not in signal_mask_patches:
            continue;

        ax.add_patch(matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha))

    if args.proj == "xy":
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
    if args.proj == "xz":
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("z", fontsize=18)
    if args.proj == "zy":
        ax.set_xlabel("z", fontsize=18)
        ax.set_ylabel("y", fontsize=18)

    # fig.tight_layout()
    ax.set_xlim(500, 900)
    ax.set_ylim(-300, 300)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("pred_points_yml", type=str)
    parser.add_argument("target_points_yml", type=str)
    parser.add_argument("in_points_yml", type=str)
    parser.add_argument("voxel_map", type=str)

    parser.add_argument("--proj", type=str, default="xy", help="xy|xz|zy")
    parser.add_argument("--adc_scalefactor", type=float, default=1.0)

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

