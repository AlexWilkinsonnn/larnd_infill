import argparse, os
from functools import partialmethod

import sparse, h5py, yaml
import numpy as np
from tqdm import tqdm

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

from aux import plot_ndlar_voxels_2

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)

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

    f = h5py.File(args.input_file, "r")

    make_voxels(
        detector,
        geometry,
        vmap,
        f["packets"], f["mc_packets_assn"], f["tracks"],
        args.output_dir, os.path.basename(args.input_file).split(".h5")[0],
        forward_facing_anode_zshift=args.forward_facing_anode_zshift,
        backward_facing_anode_zshift=args.backward_facing_anode_zshift,
        check_z_local=True,
        smear_z=args.smear_z,
        plot_only=args.plot_only,
    )

def make_voxels(
    detector,
    geometry,
    vmap,
    raw_packets, raw_mc_packets_assn, raw_tracks,
    output_dir, output_prefix,
    forward_facing_anode_zshift=0.0,
    backward_facing_anode_zshift=0.0,
    check_z_local=False,
    smear_z=1,
    plot_only=False
):
    x_bin_edges = sorted([ bin[0] for bin in vmap["x"] if type(bin) == tuple ])
    x_bin_edges.append(max(bin[1] for bin in vmap["x"] if type(bin) == tuple))
    x_bin_edges = np.array(x_bin_edges)
    y_bin_edges = sorted([ bin[0] for bin in vmap["y"] if type(bin) == tuple ])
    y_bin_edges.append(max(bin[1] for bin in vmap["y"] if type(bin) == tuple))
    y_bin_edges = np.array(y_bin_edges)
    z_bin_edges = sorted([ bin[0] for bin in vmap["z"] if type(bin) == tuple ])
    z_bin_edges.append(max(bin[1] for bin in vmap["z"] if type(bin) == tuple))
    z_bin_edges = np.array(z_bin_edges)

    packets = get_events_no_cuts(
        raw_packets, raw_mc_packets_assn, raw_tracks, geometry, detector, no_tracks=True
    )
    tracks=None
    # packets, tracks = get_events_no_cuts(
    #     f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=False
    # )

    for i_ev, event_packets in enumerate(packets):
        voxel_data = {}
        for p in event_packets:
            if p.timestamp < p.t_0:
                raise Exception("p.timestamp < p.t_0. {} and {}".format(p.timestamp, p.t_0))

            # This still happens rarely.
            # Accounting for time of interaction with LAr (~0.3us max) and diffusion (~0.02cm max)
            # does not explain some of the p.z() seen.
            # Don't understand but ignoring as for most events there are none.
            if check_z_local and p.z() >= 50.4:
                continue

            if p.io_group in [1, 2]: # means anode faces positive z direction just because it does
                z = p.z_global(centre=True) + forward_facing_anode_zshift
            else:
                z = p.z_global(centre=True) + backward_facing_anode_zshift

            coord_x = np.histogram([p.x + p.anode.tpc_x], bins=x_bin_edges)[0].nonzero()[0][0]
            coord_y = np.histogram([p.y + p.anode.tpc_y], bins=y_bin_edges)[0].nonzero()[0][0]
            coord_z = np.histogram([z], bins=z_bin_edges)[0].nonzero()[0][0]

            if p.io_group in [1, 2]:
                smear_z = min(smear_z, vmap["n_voxels"]["z"] - coord_z)
            else:
                smear_z = min(smear_z, coord_z + 1)

            for shift in range(smear_z):
                coord = (coord_x, coord_y, coord_z + (shift if p.io_group in [1, 2] else -shift))

                if coord not in voxel_data:
                    voxel_data[coord] = {}
                    voxel_data[coord]["tot_adc"] = p.ADC / smear_z
                    voxel_data[coord]["tot_packets"] = 1
                else:
                    voxel_data[coord]["tot_adc"] += p.ADC / smear_z
                    voxel_data[coord]["tot_packets"] += 1

        coords = [[], [], [], []]
        feats = []
        for coord, data in voxel_data.items():
            for i_feat, feat_name in enumerate(["tot_adc", "tot_packets"]):
                for i in range(3):
                    coords[i].append(coord[i])
                coords[3].append(i_feat)
                feats.append(data[feat_name])

        if plot_only:
            print(i_ev)
            plot_ndlar_voxels_2(
                [
                    [ coord for coord, coord_feat in zip(coords[i], coords[3]) if coord_feat == 0 ]
                    for i in range(3)
                ],
                [ feat for feat, coord_feat in zip(feats, coords[3]) if coord_feat == 0 ],
                detector, vmap["x"], vmap["y"], vmap["z"], vmap["x_gaps"], vmap["z_gaps"],
                z_scalefactor=1, max_feat=150, tracks=tracks
            )
            # plot_ndlar_voxels_2(
            #     [
            #         [ coord for coord, coord_feat in zip(coords[i], coords[3]) if coord_feat == 1 ]
            #         for i in range(3)
            #     ],
            #     [ feat for feat, coord_feat in zip(feats, coords[3]) if coord_feat == 1 ],
            #     detector, vmap["x"], vmap["y"], vmap["z"],
            #     z_scalefactor=1, max_feat=5, tracks=tracks
            # )
            continue

        s_voxelised = sparse.COO(
            coords, feats,
            shape=(x_bin_edges.size - 1, y_bin_edges.size - 1, z_bin_edges.size - 1, 2)
        )

        sparse.save_npz(
            os.path.join(output_dir, output_prefix + "_ev{}.npz".format(i_ev)), s_voxelised
        )

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument(
        "vmap", type=str,
        help="Location of generated voxelisation map to use"
    )

    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--batch_mode", action="store_true")
    # Supposed to account for self-triggering of pixels requiring current to build up. This should
    # make the packets in the z direction less sparse.
    parser.add_argument(
        "--smear_z", type=int, default=1,
        help="average z bins over n bins in the earlier time direction"
    )
    # Reconstructing the drift coordinate from ND-LAr packets always seems to put the packet a bit
    # closer than the true depo. Cannot figure out why from the code so just correcting it here.
    # Might need to change these values if using different larnd-sim/larpixsoft.
    parser.add_argument(
        "--forward_facing_anode_zshift", type=float, default=0.0,
        help="z shift to apply to all packets from a positive z facing anode (recommend +0.38cm)"
    )
    parser.add_argument(
        "--backward_facing_anode_zshift", type=float, default=0.0,
        help="z shift to apply to all packets from a negative z facing anode (recommend -0.38cm)"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.batch_mode:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    main(args)

