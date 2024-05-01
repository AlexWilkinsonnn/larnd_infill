import argparse, os
from functools import partialmethod

import sparse, h5py, yaml
import numpy as np
from tqdm import tqdm

INFILL_VTXINGAP = 0
INFILL_HADEFRACINGAP_MAX = 0.2
INFILL_LEPEFRACINGAP_MAX = 0.2

def main(args):
    with open(args.vmap, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)

    f = h5py.File(args.input_file, "r")

    if args.infillinfo_cuts:
        cut_event_ids = get_infill_cut_eventids(f)
        print(f"Cutting {len(cut_event_ids)} from file")
    else:
        cut_event_ids = set()

    make_voxels(
        f,
        vmap,
        args.output_dir, os.path.basename(args.input_file).split(".h5")[0],
        cut_event_ids=cut_event_ids,
        forward_facing_anode_zshift=args.forward_facing_anode_zshift,
        backward_facing_anode_zshift=args.backward_facing_anode_zshift,
        check_z_local=True,
        smear_z=args.smear_z
    )

def make_voxels(
    f,
    vmap,
    output_dir, output_prefix,
    cut_event_ids=set(),
    forward_facing_anode_zshift=0.0,
    backward_facing_anode_zshift=0.0,
    check_z_local=False,
    smear_z=1
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

    for event_id in np.unique(f["3d_packets"]["eventID"]):
        if event_id in cut_event_ids:
            continue

        voxel_data = {}
        for p3d in f["3d_packets"][f["3d_packets"]["eventID"] == event_id]:
            if check_z_local and p3d["z_module"] >= 50.4:
                print(
                    "!!!p.z_module = {} (>=50.4) (forward_facing_anode={}, adc={})!!!".format(
                        p3d["z_module"], p3d["forward_facing_anode"], p3d["adc"]
                    )
                )
                continue

            if p3d["forward_facing_anode"]:
                z = p3d["z"] + forward_facing_anode_zshift
            else:
                z = p3d["z"] + backward_facing_anode_zshift

            coord_x = np.histogram([p3d["x"]], bins=x_bin_edges)[0].nonzero()[0][0]
            coord_y = np.histogram([p3d["y"]], bins=y_bin_edges)[0].nonzero()[0][0]
            coord_z = np.histogram([z], bins=z_bin_edges)[0].nonzero()[0][0]

            if p3d["forward_facing_anode"]:
                smear_z = min(smear_z, vmap["n_voxels"]["z"] - coord_z)
            else:
                smear_z = min(smear_z, coord_z + 1)

            for shift in range(smear_z):
                coord = (
                    coord_x, coord_y, coord_z + (shift if p3d["forward_facing_anode"] else -shift)
                )

                if coord not in voxel_data:
                    voxel_data[coord] = {}
                    voxel_data[coord]["tot_adc"] = p3d["adc"] / smear_z
                    voxel_data[coord]["tot_packets"] = 1
                else:
                    voxel_data[coord]["tot_adc"] += p3d["adc"] / smear_z
                    voxel_data[coord]["tot_packets"] += 1

        coords = [[], [], [], []]
        feats = []
        for coord, data in voxel_data.items():
            for i_feat, feat_name in enumerate(["tot_adc", "tot_packets"]):
                for i in range(3):
                    coords[i].append(coord[i])
                coords[3].append(i_feat)
                feats.append(data[feat_name])

        s_voxelised = sparse.COO(
            coords, feats,
            shape=(x_bin_edges.size - 1, y_bin_edges.size - 1, z_bin_edges.size - 1, 2)
        )

        sparse.save_npz(
            os.path.join(output_dir, output_prefix + "{}.npz".format(event_id)), s_voxelised
        )

def get_infill_cut_eventids(f):
    mask = (f["nd_paramreco"]["vtxInGap"] == INFILL_VTXINGAP)
    mask = mask & (f["nd_paramreco"]["hadEFracInGap"] < INFILL_HADEFRACINGAP_MAX)
    mask = mask & (f["nd_paramreco"]["lepEFracInGap"] < INFILL_LEPEFRACINGAP_MAX)
    return set(f["nd_paramreco"][~mask]["eventID"])

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
    parser.add_argument(
        "--infillinfo_cuts", action="store_true",
        help="Apply hardcoded cuts based on infill info stored in nd_paramreco"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.batch_mode:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    main(args)

