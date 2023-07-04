import argparse, os, math

import sparse, h5py
import numpy as np
from matplotlib import pyplot as plt

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

from aux import plot_ndlar_voxels

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
PIXEL_COL_OFFSET = 128
PIXEL_COLS_PER_ANODE = 256
PIXEL_COLS_PER_GAP = 11 # 4.14 / 0.38
PIXEL_ROW_OFFSET = 405
TICK_OFFSET = 0
TICKS_PER_MODULE = 6116
TICKS_PER_GAP = 79 # 1.3cm / (0.1us * 0.1648cm/us)


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    f = h5py.File(args.input_file, "r")

    # print(detector.tpc_borders)
    # print(detector.pixel_pitch)
    # print(detector.vdrift)
    # print(detector.time_sampling)
    # print((356.7 - 255.9) / (0.1 * 0.1648))
    # print(detector.tpc_offsets)
    # import sys; sys.exit()

    # xs, ys = [], []
    # for x, y in geometry.values():
    #     xs.append(math.floor(x / detector.pixel_pitch))
    #     ys.append(math.floor(y / detector.pixel_pitch))
    # print(min(xs), max(xs))
    # print(min(ys), max(ys))
    # import sys; sys.exit()
    # -128 127 (x)
    # -405 394 (y)

    packets = get_events_no_cuts(
        f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=True
    )

    tpc_offsets_x = sorted(list(set(offsets[0] for offsets in detector.tpc_offsets )))
    tpc_offsets_z = sorted(list(set(offsets[2] for offsets in detector.tpc_offsets )))

    for event_packets in packets:
        coords = [[], [], []]
        adcs = []
        num_before_trigger, num_zero_adc, num_high_z = 0, 0, 0
        for p in event_packets:
            # Very rarely some packets are one tick before the trigger packet,
            # not sure what causes this
            if p.timestamp < p.t_0:
                num_before_trigger += 1
                continue

            if p.ADC == 0:
                num_zero_adc += 1

            # This is either caused by longitudinal diffusion, or the interactions with the LAr
            # taking a non negligible amount of time ie. not being instantaneous with the t_0 flash
            if p.z() > 50.4:
                num_high_z += 1
                continue

            voxel_x = math.floor(p.x / detector.pixel_pitch)
            voxel_x += PIXEL_COL_OFFSET
            voxel_x += (
                tpc_offsets_x.index(p.anode.tpc_x) * (PIXEL_COLS_PER_ANODE + PIXEL_COLS_PER_GAP)
            )
            coords[0].append(voxel_x)

            voxel_y = math.floor(p.y / detector.pixel_pitch)
            voxel_y += PIXEL_ROW_OFFSET
            coords[1].append(voxel_y)

            voxel_z = math.floor(p.z() / (detector.time_sampling * detector.vdrift))
            voxel_z += TICK_OFFSET
            voxel_z += tpc_offsets_z.index(p.anode.tpc_z) * (TICKS_PER_MODULE + TICKS_PER_GAP)
            coords[2].append(voxel_z)

            adcs.append(p.ADC)

        plot_ndlar_voxels(
            coords, adcs, detector,
            pix_cols_per_anode=PIXEL_COLS_PER_ANODE, pix_cols_per_gap=PIXEL_COLS_PER_GAP,
            pix_rows_per_anode=800,
            ticks_per_module=TICKS_PER_MODULE, ticks_per_gap=TICKS_PER_GAP
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

