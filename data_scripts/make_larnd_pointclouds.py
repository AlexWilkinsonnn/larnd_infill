import argparse, os

import sparse, h5py
import numpy as np
from matplotlib import pyplot as plt

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    f = h5py.File(args.input_file, "r")

    packets, _ = get_events_no_cuts(
        f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector
    )

    for event_packets in packets:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        packet_x, packet_y, packet_z = [], [], []
        for p in event_packets:
            packet_x.append(p.x + p.anode.tpc_x)
            packet_y.append(p.y + p.anode.tpc_y)
            packet_z.append(p.z_global())

        ax.scatter(packet_z, packet_x, packet_y, marker='o')

        xlims = [413.72, 916.68]
        ylims = [-148.613, 155.387]
        zlims = [-356.7, 356.7]
        lines = [
            ((xlims[0], xlims[1]), (zlims[0],) * 2, (ylims[0],) * 2), # left low /
            ((xlims[0],) * 2, (zlims[0], zlims[1]), (ylims[0],) * 2), # low front -
            ((xlims[0],) * 2, (zlims[0],) * 2, (ylims[0], ylims[1])), # left front |
            ((xlims[0], xlims[1]), (zlims[0],) * 2, (ylims[1],) * 2), # left up /
            ((xlims[0],) * 2, (zlims[0], zlims[1]), (ylims[1],) * 2), # up front -
            ((xlims[1],) * 2, (zlims[0], zlims[1]), (ylims[1],) * 2), # up back -
            ((xlims[1],) * 2, (zlims[0],) * 2, (ylims[1], ylims[0])), # left back |
            ((xlims[0],) * 2, (zlims[1],) * 2, (ylims[0], ylims[1])), # right front |
            ((xlims[0], xlims[1]), (zlims[1],) * 2, (ylims[0],) * 2), # right low /
            ((xlims[1],) * 2, (zlims[1],) * 2, (ylims[0], ylims[1])), # right back |
            ((xlims[1],) * 2, (zlims[1], zlims[0]), (ylims[0],) * 2), # low back -
            ((xlims[0], xlims[1]), (zlims[1],) * 2, (ylims[1],) * 2)  # right up /
        ]
        for line in lines:
            ax.plot(line[1], line[0], zs=line[2], color="black")

        ax.set_box_aspect((1,1,1))

        ax.axes.set_xlim3d(left=-400, right=400)
        ax.axes.set_ylim3d(bottom=400, top=1000)
        ax.axes.set_zlim3d(bottom=-200, top=200)

        fig.tight_layout()
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

