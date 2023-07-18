import argparse, os

import sparse, h5py, yaml
import numpy as np

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

from aux import plot_ndlar_voxels, plot_ndlar_voxels_2

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)
# DET_PROPS=(
#     "/home/alex/Documents/extrapolation/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
# )
# PIXEL_LAYOUT=(
#     "/home/alex/Documents/extrapolation/larnd-sim/larndsim/pixel_layouts/"
#     "multi_tile_layout-3.0.40.yaml"
# )


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    with open(args.vmap, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)

    assertion = detector.vdrift == vmap["vdrift"]
    assert assertion, "vdrift mismatch {} and {}".format(detector.vdrift, vmap["vdrift"])
    assertion = detector.pixel_pitch == vmap["pixel_pitch"]
    message = "pixel_pitch mismatch {} and {}".format(detector.pixel_pitch, vmap["pixel_pitch"])
    assert assertion, message

    x_bin_edges = sorted([ bin[0] for bin in vmap["x"] if type(bin) == tuple ])
    x_bin_edges.append(max(bin[1] for bin in vmap["x"] if type(bin) == tuple))
    x_bin_edges = np.array(x_bin_edges)
    y_bin_edges = sorted([ bin[0] for bin in vmap["y"] if type(bin) == tuple ])
    y_bin_edges.append(max(bin[1] for bin in vmap["y"] if type(bin) == tuple))
    y_bin_edges = np.array(y_bin_edges)
    z_bin_edges = sorted([ bin[0] for bin in vmap["z"] if type(bin) == tuple ])
    z_bin_edges.append(max(bin[1] for bin in vmap["z"] if type(bin) == tuple))
    z_bin_edges = np.array(z_bin_edges)

    z_bin_width = vmap["z_step"]

    f = h5py.File(args.input_file, "r")

    packets = get_events_no_cuts(
        f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=True
    )

    for i_ev, event_packets in enumerate(packets):
        voxel_data = {}
        for p in event_packets:
            if p.timestamp < p.t_0:
                raise Exception("p.timestamp < p.t_0. {} and {}".format(p.timestamp, p.t_0))

            # if p.ADC == 0:
            #     continue

            if p.z() > 50.4:
                raise Exception("p.z() > 50.4. {}".format(p.z()))

            coord_x = np.histogram([p.x + p.anode.tpc_x], bins=x_bin_edges)[0].nonzero()[0][0]
            coord_y = np.histogram([p.y + p.anode.tpc_y], bins=y_bin_edges)[0].nonzero()[0][0]
            coord_z = np.histogram(
                [p.z_global() + z_bin_width / 2], bins=z_bin_edges
            )[0].nonzero()[0][0]

            coord = (coord_x, coord_y, coord_z)

            if coord not in voxel_data:
                voxel_data[coord] = {}
                voxel_data[coord]["tot_adc"] = p.ADC
                voxel_data[coord]["tot_packets"] = 1
            else:
                voxel_data[coord]["tot_adc"] += p.ADC
                voxel_data[coord]["tot_packets"] += 1

        coords = [[], [], [], []]
        feats = []
        for coord, data in voxel_data.items():
            for i_feat, feat_name in enumerate(["tot_adc", "tot_packets"]):
                for i in range(3):
                    coords[i].append(coord[i])
                coords[3].append(i_feat)
                feats.append(data[feat_name])

        if args.plot_only:
            # plot_ndlar_voxels_2(
            #     coords, [ feat[0] for feat in feats ], detector, vmap["x"], vmap["y"], vmap["z"],
            #     z_scalefactor=1, max_feat=300
            # )
            plot_ndlar_voxels_2(
                coords, [ feat[1] for feat in feats ], detector, vmap["x"], vmap["y"], vmap["z"],
                z_scalefactor=1, max_feat=20
            )
            continue

        s_voxelised = sparse.COO(
            coords, feats,
            shape=(x_bin_edges.size - 1, y_bin_edges.size - 1, z_bin_edges.size - 1, 2)
        )

        sparse.save_npz(
            os.path.join(
                args.output_dir,
                os.path.basename(args.input_file).split(".h5")[0] + "_ev{}.npz".format(i_ev)
            ),
            s_voxelised
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("vmap", type=str, help="Location of generated voxelisation map to use")

    parser.add_argument("--plot_only", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

