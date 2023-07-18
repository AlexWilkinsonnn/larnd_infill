import argparse

import h5py

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

from aux import plot_ndlar

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

    if args.tracks:
        packets, tracks = get_events_no_cuts(
            f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector
        )
        for event_packets, event_tracks in zip(packets, tracks):
            plot_ndlar(event_packets, detector, tracks=event_tracks)
    else:
        packets = get_events_no_cuts(
            f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=True
        )
        for event_packets in packets:
            # plot_ndlar(event_packets, detector)
            plot_ndlar(event_packets, detector, projections=True, structures=False)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")

    parser.add_argument("--tracks", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

