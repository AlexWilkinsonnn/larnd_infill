import argparse, os

import sparse, h5py
import numpy as np
from matplotlib import pyplot as plt

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

# DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
# PIXEL_LAYOUT=(
#     "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
# )
DET_PROPS=(
    "/home/alex/Documents/extrapolation/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
)
PIXEL_LAYOUT=(
    "/home/alex/Documents/extrapolation/larnd-sim/larndsim/pixel_layouts/"
    "multi_tile_layout-3.0.40.yaml"
)


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    f = h5py.File(args.input_file, "r")

    packets = get_events_no_cuts(
        f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=True
    )

    # voxel number will be for x:
    # pixel column number + num previous drift modules * (num pixel columns per drift module * num pixel columns per drift module side gap)
    # voxel number will be for y:
    # pixel row number
    # voxel number will be for z:
    # tick number + num previous drift modules * (num ticks per drift module * num ticks per anode gap)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

