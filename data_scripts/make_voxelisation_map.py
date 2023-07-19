"""
Make bi-directional dictionaries (dict[1] = 10; dict[10] = 1) for the mapping of voxel coordinates
to detector coordinates. Detector coordinates are bin edges like [low, high)
"""
import argparse

import numpy as np
import yaml

from larpixsoft.detector import set_detector_properties

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)
RES=7 # cm at 7 dp (nm)


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    if args.preset == 0:
        x_step, y_step = detector.pixel_pitch, detector.pixel_pitch
        z_step = detector.time_sampling * detector.vdrift
    elif args.preset == 1:
        x_step, y_step = detector.pixel_pitch, detector.pixel_pitch
        z_step =  detector.vdrift
    else:
        raise ValueError("No preset={}".format(args.preset))

    voxel_maps = {
        "preset" : args.preset,
        "x_step_target" : x_step, "y_step_target" : y_step, "z_step_target" : z_step,
        "vdrift" : detector.vdrift,
        "pixel_pitch" : detector.pixel_pitch
    }

    voxel_maps["x"], voxel_maps["x_gaps"] = voxelise_x(detector.tpc_borders[:, 0, :], x_step)
    voxel_maps["y"], voxel_maps["y_gaps"] = voxelise_y(detector.tpc_borders[:, 1, :], y_step)
    voxel_maps["z"], voxel_maps["z_gaps"] = voxelise_z(detector.tpc_borders[:, 2, :], z_step)

    voxel_maps["n_voxels"] = {}
    n_voxels_tot = 1
    print("Number of voxels")
    for coord in ["x", "y", "z"]:
        n_voxels = len([ key for key in voxel_maps[coord] if type(key) == tuple ])
        n_voxels_tot *= n_voxels
        print("{}: {}".format(coord, n_voxels))
        voxel_maps["n_voxels"][coord] = n_voxels
    print("Total: {}\n".format(n_voxels_tot))

    print("Gaps at voxels:")
    print("x: {}".format(voxel_maps["x_gaps"]))
    print("y: {}".format(voxel_maps["y_gaps"]))
    print("z: {}".format(voxel_maps["z_gaps"]))

    with open(args.out_path, "w") as f:
        yaml.dump(voxel_maps, f)


def voxelise_y(borders, step):
    borders = np.unique(borders.round(decimals=RES))

    assert borders.size == 2, "Expected all modules to have the same y range: {}".format(borders)

    min_coord, max_coord = np.min(borders), np.max(borders)
    bins = np.linspace(min_coord, max_coord, int((max_coord - min_coord) / step) + 1)

    voxel_map = {}
    gap_voxels = []
    add_bins_to_voxel_map(voxel_map, bins)

    return voxel_map, gap_voxels


def voxelise_x(borders, step):
    borders = np.unique(borders.round(decimals=RES))

    gap_sizes = np.diff(borders)[1::2]
    assertion = np.unique(gap_sizes.round(decimals=RES)).size == 1
    assert assertion, "Expected equal x gap sizes: {}".format(gap_sizes)
    gap_size = gap_sizes[0]

    module_sizes = np.diff(borders)[::2]
    assertion = np.unique(module_sizes.round(decimals=RES)).size == 1
    assert assertion, "Expected equal x module sizes: {}".format(module_sizes)
    module_size = module_sizes[0]

    n_bins_module = round(module_size / step)
    n_bins_gap = round(gap_size / step)

    voxel_map = {}
    gap_voxels = []
    i_bin_start = 0
    for border_l, border_u, border_l_next in zip(
        borders[::2], borders[1::2], np.append(borders, None)[2::2]
    ):
        bins = np.linspace(border_l, border_u, n_bins_module + 1)
        add_bins_to_voxel_map(voxel_map, bins, i_bin_start=i_bin_start)
        i_bin_start += bins.size - 1
        if border_l_next is not None:
            bins = np.linspace(border_u, border_l_next, n_bins_gap + 1)

            for gap_voxel in range(i_bin_start, i_bin_start + bins.size - 1):
                gap_voxels.append(gap_voxel)

            add_bins_to_voxel_map(voxel_map, bins, i_bin_start=i_bin_start)
            i_bin_start += bins.size - 1

    return voxel_map, gap_voxels


def voxelise_z(borders, step):
    # anode - cathode - anode - (gap) - anode - ...
    borders = np.unique(borders.round(decimals=RES))
    cathode_mask = np.ones(borders.size, dtype=bool)
    cathode_mask[1::3] = 0
    borders = borders[cathode_mask]

    gap_sizes = np.diff(borders)[1::2]
    assertion = np.unique(gap_sizes.round(decimals=RES)).size == 1
    assert assertion, "Expected equal z gap sizes: {}".format(gap_sizes)
    gap_size = gap_sizes[0]

    module_sizes = np.diff(borders)[::2]
    assertion = np.unique(module_sizes.round(decimals=RES)).size == 1
    assert assertion, "Expected equal z module sizes: {}".format(module_sizes)
    module_size = module_sizes[0]

    n_bins_module = round(module_size / step)
    n_bins_gap = round(gap_size / step)

    voxel_map = {}
    gap_voxels = []
    i_bin_start = 0
    for border_l, border_u, border_l_next in zip(
        borders[::2], borders[1::2], np.append(borders, None)[2::2]
    ):
        bins = np.linspace(border_l, border_u, n_bins_module + 1)
        add_bins_to_voxel_map(voxel_map, bins, i_bin_start=i_bin_start)
        i_bin_start += bins.size - 1
        if border_l_next is not None:
            bins = np.linspace(border_u, border_l_next, n_bins_gap + 1)

            for gap_voxel in range(i_bin_start, i_bin_start + bins.size - 1):
                gap_voxels.append(gap_voxel)

            add_bins_to_voxel_map(voxel_map, bins, i_bin_start=i_bin_start)
            i_bin_start += bins.size - 1

    return voxel_map, gap_voxels


def add_bins_to_voxel_map(voxel_map, bins, i_bin_start=0):
    for i_bin, (bin_l, bin_u) in enumerate(zip(bins[:-1], bins[1:])):
        binl_binu = (round(float(bin_l), RES), round(float(bin_u), RES))
        voxel_map[i_bin + i_bin_start] = binl_binu
        voxel_map[binl_binu] = i_bin + i_bin_start


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("out_path", type=str)
    parser.add_argument("preset", type=int, help="0 (no downsampling)|1 (downsample z by 10)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

