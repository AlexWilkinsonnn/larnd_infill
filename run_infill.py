"""
Load larnd-sim data from h5 file and run trained infill model on it.
- Load packets, prepare as voxel data
- Pass to dataloader to make infill mask
- Apply infill network
- Map back to unvoxelised 3d positions, keeping the non-infilled packets unchanged
NOTE: The zshifting is applied here before the infill step. So it should be be applied again
when loading into larsoft
"""
import argparse, os, shutil
import datetime

import h5py, sparse
from tqdm import tqdm
import numpy as np

import torch

from ME.config_parsers.parser_eval import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net_adversarial import CompletionNetAdversarialEval

def main(args, overwrite_dict):
    conf = get_config(args.config, overwrite_dict=overwrite_dict)

    in_f = h5py.File(conf.input_file, "r")

    # Make voxel data from packets and write to disk
    voxels_dir = os.path.join(
        conf.cache_dir, datetime.datetime.now().strftime("%y-%m-%d_%H_%M_%S")
    )
    print("Disk cache for voxel data is {}".format(voxels_dir))
    if not os.path.exists(voxels_dir):
        os.makedirs(voxels_dir)
    print("Making voxels...")
    make_voxels(conf, in_f, voxels_dir)

    model = CompletionNetAdversarialEval(conf)

    collate_fn = CollateCOO(
        coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    )
    dataset = LarndDataset(
        voxels_dir,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        conf.xyz_smear_infill, conf.xyz_smear_active,
        conf.xyz_max_reflect_distance,
        seed=1
    )
    dataset.set_use_true_gaps(True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, collate_fn=collate_fn, num_workers=0
    )

    with h5py.File(args.output_file, "w") as out_f:
        for key in in_f.keys():
            data = np.array(in_f[key])
            out_f.create_dataset(key, data=data)
        p3d_infill_dtype = in_f["3d_packets"].dtype
        out_f.create_dataset("3d_packets_infilled", (0,), dtype=p3d_infill_dtype, maxshape=(None,))

        packets_3d_infill_list = []

        for data in tqdm(dataloader, desc="Running Infill Network"):
            model.set_input(data)
            model.test()
            vis = model.get_current_visuals()
            s_in, s_pred = vis["s_in"], vis["s_pred"]
            s_in_infill_mask = s_in.F[:, -1] == 1
            infill_coords = s_in.C[s_in_infill_mask].type(torch.float)

            b_size = len(data["mask_x"])
            for i_b in range(b_size):
                event_id = int(os.path.basename(data["data_path"][i_b]).rstrip(".npz"))

                orig_p3d = in_f["3d_packets"][in_f["3d_packets"]["eventID"] == event_id]

                infill_coords_b = infill_coords[infill_coords[:, 0] == i_b]
                infill_feats_b = s_pred.features_at_coordinates(infill_coords_b)
                infill_feats_b_nonzero_mask = infill_feats_b[:, 0] != 0
                infill_coords_b = infill_coords_b[infill_feats_b_nonzero_mask].type(torch.int)
                infill_feats_b = infill_feats_b[infill_feats_b_nonzero_mask]
                infill_feats_b = infill_feats_b / conf.scalefactors[0]

                packets_3d_infill_ev = np.empty(
                    len(orig_p3d) + len(infill_coords_b), dtype=p3d_infill_dtype
                )
                for i_p, p3d in enumerate(orig_p3d):
                    packets_3d_infill_ev[i_p] = p3d
                for i_p, (coord, feat) in enumerate(zip(infill_coords_b, infill_feats_b)):
                    x_l_h = conf.vmap["x"][coord[1].item()]
                    x = (x_l_h[0] + x_l_h[1]) / 2
                    y_l_h = conf.vmap["y"][coord[2].item()]
                    y = (y_l_h[0] + y_l_h[1]) / 2
                    z_l_h = conf.vmap["z"][coord[3].item()]
                    z = (z_l_h[0] + z_l_h[1]) / 2
                    i_p += len(orig_p3d)
                    packets_3d_infill_ev[i_p]["eventID"] = event_id
                    packets_3d_infill_ev[i_p]["adc"] = feat.item()
                    packets_3d_infill_ev[i_p]["x"] = x
                    packets_3d_infill_ev[i_p]["x_module"] = 0
                    packets_3d_infill_ev[i_p]["y"] = y
                    packets_3d_infill_ev[i_p]["z"] = z
                    packets_3d_infill_ev[i_p]["z_module"] = 0
                    packets_3d_infill_ev[i_p]["forward_facing_anode"] = 2
                    packets_3d_infill_ev[i_p]["infilled"] = 1
                packets_3d_infill_list.append(packets_3d_infill_ev)

        packets_3d_infill = np.concatenate(packets_3d_infill_list, axis=0)
        out_f["3d_packets_infilled"].resize((len(packets_3d_infill),))
        out_f["3d_packets_infilled"][:] = packets_3d_infill

    print("Removing cache dir {}".format(voxels_dir))
    shutil.rmtree(voxels_dir)

def make_voxels(conf, f, output_dir):
    """
    Voxel sparse data from larnd-sim packets that have been reconstructed to 3d positions.
    Files named using eventid.
    """
    x_bin_edges = sorted([ bin[0] for bin in conf.vmap["x"] if type(bin) == tuple ])
    x_bin_edges.append(max(bin[1] for bin in conf.vmap["x"] if type(bin) == tuple))
    x_bin_edges = np.array(x_bin_edges)
    y_bin_edges = sorted([ bin[0] for bin in conf.vmap["y"] if type(bin) == tuple ])
    y_bin_edges.append(max(bin[1] for bin in conf.vmap["y"] if type(bin) == tuple))
    y_bin_edges = np.array(y_bin_edges)
    z_bin_edges = sorted([ bin[0] for bin in conf.vmap["z"] if type(bin) == tuple ])
    z_bin_edges.append(max(bin[1] for bin in conf.vmap["z"] if type(bin) == tuple))
    z_bin_edges = np.array(z_bin_edges)

    for event_id in np.unique(f["3d_packets"]["eventID"]):
        voxel_data = {}
        for p3d in f["3d_packets"][f["3d_packets"]["eventID"] == event_id]:
            if p3d["z_module"] >= 50.4:
                print(
                    "!!!p.z_module = {} (>=50.4) (forward_facing_anode={})!!!".format(
                        p3d["z_module"], p3d["forward_facing_anode"]
                    )
                )
                continue

            if p3d["forward_facing_anode"]:
                z = p3d["z"] + conf.forward_facing_anode_zshift
            else:
                z = p3d["z"] + conf.backward_facing_anode_zshift

            coord_x = np.histogram([p3d["x"]], bins=x_bin_edges)[0].nonzero()[0][0]
            coord_y = np.histogram([p3d["y"]], bins=y_bin_edges)[0].nonzero()[0][0]
            coord_z = np.histogram([z], bins=z_bin_edges)[0].nonzero()[0][0]

            if p3d["forward_facing_anode"]:
                smear_z = min(conf.smear_z, conf.vmap["n_voxels"]["z"] - coord_z)
            else:
                smear_z = min(conf.smear_z, coord_z + 1)

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

        sparse.save_npz(os.path.join(output_dir, "{}.npz".format(event_id)), s_voxelised)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str)

    parser.add_argument(
        "--input_file", type=str, default=None, help="override config input_file"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="override config output_file"
    )

    args = parser.parse_args()

    overwrite_dict = {}
    if args.input_file is not None:
        overwrite_dict["input_file"] = args.input_file
    if args.output_file is not None:
        overwrite_dict["output_file"] = args.output_file

    return args, overwrite_dict

if __name__ == "__main__":
    args, overwrite_dict = parse_arguments()
    main(args, overwrite_dict)
