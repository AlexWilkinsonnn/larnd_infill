"""
Make bi-directional dictionaries (dict[1] = 10; dict[10] = 1) for the mapping of voxel coordinates
to detector coordinates. Detector coordinates are bin edges like [low, high)
"""
import argparse

import numpy as np
import yaml

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    y_voxel_map = voxelise_y(detector.tpc_borders[:, 1, :], detector.pixel_pitch)

    x_voxel_map = voxelise_x(detector.tpc_borders[:, 0, :], detector.pixel_pitch)

    if args.type == 0:
        pass
        
        

    elif args.type == 1:
        pass
        

def voxelise_y(borders, step):
    borders = np.array(
        sorted({ border for low_high_borders in borders for border in low_high_borders })
    )

    assert borders.size == 2, "Expected all modules to have the same y range"

    min_coord, max_coord = np.min(borders), np.max(borders)
    bins = np.linspace(min_coord, max_coord, int((max_coord - min_coord) / step) + 1)

    voxel_map = {}
    for i_bin, (bin_l, bin_u) in enumerate(zip(bins[:-1], bins[1:])):
        voxel_map[i_bin] = (bin_l, bin_u)
        voxel_map[(bin_l, bin_u)] = i_bin

    return voxel_map


def voxelise_x(borders, step):
    voxel_map = {}

    borders = np.array(
        sorted({ border for low_high_borders in borders for border in low_high_borders })
    )

    gap_sizes = np.diff(borders)[1::2]
    assertion = all(
        np.isclose(gap_size_1, gap_size_2) for gap_size_2 in gap_sizes for gap_size_1 in gap_sizes
    )
    assert assertion, "Expected equal x gap sizes"
    gap_size = gap_sizes[0]

    module_sizes = np.diff(borders)[::2]
    assertion = all(
        np.isclose(module_size_1, module_size_2)
        for module_size_2 in module_sizes
            for module_size_1 in module_sizes
    )
    assert assertion, "Expected equal x module sizes"
    module_size = module_sizes[0]

    n_bins_module = round(module_size / step)
    n_bins_gap = round(gap_size / step)
    print(n_bins_module, n_bins_gap)

    return voxel_map





def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("type", help="0 (no downsampling)|1 (downsample z by 10)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

