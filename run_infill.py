"""
Load larnd-sim data from h5 file and run trained infill model on it.
- Load packets, prepare as voxel data
- Pass to dataloader to make infill mask
- Apply infill network
- Map back to unvoxelised 3d positions, keeping the non-infilled packets unchanged
NOTE: The zshifting is applied here before the infill step. So it should not be applied again
when loading into larsoft
"""
import argparse, os, shutil, sys
import datetime

import h5py, sparse, yaml
from tqdm import tqdm
import numpy as np
import matplotlib; from matplotlib import pyplot as plt

import torch

from ME.config_parsers.parser_eval import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net_adversarial import CompletionNetAdversarialEval
from ME.train.train_sigmask_adversarial import plot_pred

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

            if args.plot_only:
                plot_pred(
                    vis["s_pred"], vis["s_in"], vis["s_target"],
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    "example_voxels",
                    conf.detector,
                    save_dir="/home/awilkins/larnd_infill/larnd_infill/infill_debug_plots",
                    save_tensors=False,
                    max_evs=conf.batch_size,
                    skip_target=True,
                    skip_predonly=True,
                    autocrop=True,
                    adc_threshold=0
                )
                print("Plotting batch model voxels finished")

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
                    len(orig_p3d) + len(infill_coords_b) * conf.infilled_voxel_splits,
                    dtype=p3d_infill_dtype
                )
                for i_p, p3d in enumerate(orig_p3d):
                    packets_3d_infill_ev[i_p]["eventID"] = event_id
                    packets_3d_infill_ev[i_p]["adc"] = p3d["adc"]
                    packets_3d_infill_ev[i_p]["x"] = p3d["x"]
                    packets_3d_infill_ev[i_p]["x_module"] = p3d["x_module"]
                    packets_3d_infill_ev[i_p]["y"] = p3d["y"]
                    if p3d["forward_facing_anode"]:
                        packets_3d_infill_ev[i_p]["z"] = (
                            p3d["z"] + conf.forward_facing_anode_zshift
                        )
                        packets_3d_infill_ev[i_p]["z_module"] = (
                            p3d["z_module"] + conf.forward_facing_anode_zshift
                        )
                    else:
                        packets_3d_infill_ev[i_p]["z"] = (
                            p3d["z"] + conf.backward_facing_anode_zshift
                        )
                        packets_3d_infill_ev[i_p]["z_module"] = (
                            p3d["z_module"] + conf.backward_facing_anode_zshift
                        )
                    packets_3d_infill_ev[i_p]["forward_facing_anode"] = p3d["forward_facing_anode"]
                    packets_3d_infill_ev[i_p]["infilled"] = 0
                i_p = len(orig_p3d)
                for coord, feat in zip(infill_coords_b, infill_feats_b):
                    x_l_h = conf.vmap["x"][coord[1].item()]
                    x = (x_l_h[0] + x_l_h[1]) / 2
                    y_l_h = conf.vmap["y"][coord[2].item()]
                    y = (y_l_h[0] + y_l_h[1]) / 2
                    z_l_h = conf.vmap["z"][coord[3].item()]
                    adc = feat.item() / conf.infilled_voxel_splits
                    for i in range(conf.infilled_voxel_splits):
                        z = (
                            z_l_h[0] +
                            (i + 1) * (z_l_h[1] - z_l_h[0]) / (conf.infilled_voxel_splits + 1)
                        )
                        packets_3d_infill_ev[i_p]["eventID"] = event_id
                        packets_3d_infill_ev[i_p]["adc"] = adc
                        packets_3d_infill_ev[i_p]["x"] = x
                        packets_3d_infill_ev[i_p]["x_module"] = 0
                        packets_3d_infill_ev[i_p]["y"] = y
                        packets_3d_infill_ev[i_p]["z"] = z
                        packets_3d_infill_ev[i_p]["z_module"] = 0
                        packets_3d_infill_ev[i_p]["forward_facing_anode"] = 2
                        packets_3d_infill_ev[i_p]["infilled"] = 1
                        i_p += 1
                packets_3d_infill_list.append(packets_3d_infill_ev)

            if args.plot_only:
                with open("voxel_maps/vmap_zdownresolution5.yml", "r") as f:
                    vmap = yaml.load(f, Loader=yaml.FullLoader)
                plot_infilled_revoxelise(
                    packets_3d_infill_list,
                    conf,
                    vmap,
                    data["mask_x"][0], data["mask_z"][0],
                    "/home/awilkins/larnd_infill/larnd_infill/infill_debug_plots",
                    "example_revoxelised"
                )
                print("Plotting batch outputted voxels finished. Exiting.")
                sys.exit()

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

def plot_infilled_revoxelise(
    p3ds_infill_list, conf, vmap, x_gaps, z_gaps, save_dir, save_name_prefix
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

    norm_feats = matplotlib.colors.Normalize(vmin=0, vmax=150)
    m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.jet)

    for i_p3ds, p3ds in enumerate(p3ds_infill_list):
        fig, ax = plt.subplots(1, 3, figsize=(24, 6))

        y_pos = conf.detector.tpc_borders[-1][1][1]
        y_size = conf.detector.tpc_borders[0][1][0] - conf.detector.tpc_borders[-1][1][1]
        z_pos = conf.detector.tpc_borders[-1][2][0]
        z_size = conf.detector.tpc_borders[0][2][0] - conf.detector.tpc_borders[-1][2][0]
        for x_gap_coord in x_gaps:
            x_bin = vmap["x"][x_gap_coord]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            ax[0].add_patch(
                matplotlib.patches.Rectangle((x_pos, y_pos), x_size, y_size, fc="gray", alpha=0.3)
            )
            ax[1].add_patch(
                matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="gray", alpha=0.3)
            )
        x_pos = conf.detector.tpc_borders[-1][0][1]
        x_size = conf.detector.tpc_borders[0][0][0] - conf.detector.tpc_borders[-1][0][1]
        y_pos = conf.detector.tpc_borders[-1][1][1]
        y_size = conf.detector.tpc_borders[0][1][0] - conf.detector.tpc_borders[-1][1][1]
        for z_gap_coord in z_gaps:
            z_bin = vmap["z"][z_gap_coord]
            z_size, z_pos = z_bin[1] - z_bin[0], z_bin[0]
            ax[1].add_patch(
                matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="gray", alpha=0.3)
            )
            ax[2].add_patch(
                matplotlib.patches.Rectangle((z_pos, y_pos), z_size, y_size, fc="gray", alpha=0.3)
            )

        coords = [
            [ np.histogram([x], bins=x_bin_edges)[0].nonzero()[0][0] for x in p3ds["x"] ],
            [ np.histogram([y], bins=y_bin_edges)[0].nonzero()[0][0] for y in p3ds["y"] ],
            [ np.histogram([z], bins=z_bin_edges)[0].nonzero()[0][0] for z in p3ds["z"] ]
        ]
        feats = [ f for f in p3ds["adc"] ]

        curr_patches_xy, curr_patches_xz, curr_patches_zy = set(), set(), set()
        for coord_x, coord_y, coord_z, feat in zip(*coords, feats):
            x_bin = vmap["x"][coord_x]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            y_bin = vmap["y"][coord_y]
            y_size, y_pos = (y_bin[1] - y_bin[0]), y_bin[0]
            z_bin = vmap["z"][coord_z]
            z_size, z_pos = (z_bin[1] - z_bin[0]), z_bin[0]

            c = m_feats.to_rgba(feat)
            alpha = 1.0

            pos_xy = (x_pos, y_pos)
            if pos_xy not in curr_patches_xy:
                curr_patches_xy.add(pos_xy)
                ax[0].add_patch(
                    matplotlib.patches.Rectangle(pos_xy, x_size, y_size, fc=c, alpha=alpha)
                )
            pos_xz = (x_pos, z_pos)
            if pos_xz not in curr_patches_xz:
                curr_patches_xz.add(pos_xz)
                ax[1].add_patch(
                    matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
                )
            pos_zy = (z_pos, y_pos)
            if pos_zy not in curr_patches_zy:
                curr_patches_zy.add(pos_zy)
                ax[2].add_patch(
                    matplotlib.patches.Rectangle(pos_zy, z_size, y_size, fc=c, alpha=alpha)
                )

        max_x = min(vmap["x"][max(coords[0])][0] + 20, conf.detector.tpc_borders[-1][0][1])
        min_x = max(vmap["x"][min(coords[0])][0] - 20, conf.detector.tpc_borders[0][0][0])
        max_y = min(vmap["y"][max(coords[1])][0] + 20, conf.detector.tpc_borders[-1][1][1])
        min_y = max(vmap["y"][min(coords[1])][0] - 20, conf.detector.tpc_borders[0][1][0])
        max_z = min(vmap["z"][max(coords[2])][0] + 20, conf.detector.tpc_borders[-1][2][0])
        min_z = max(vmap["z"][min(coords[2])][0] - 20, conf.detector.tpc_borders[0][2][0])

        ax[0].set_xlim(min_x, max_x)
        ax[1].set_xlim(min_x, max_x)
        ax[2].set_xlim(min_z, max_z)
        ax[0].set_ylim(min_y, max_y)
        ax[1].set_ylim(min_z, max_z)
        ax[2].set_ylim(min_y, max_y)
        ax[0].set_xlabel("x")
        ax[1].set_xlabel("x")
        ax[2].set_xlabel("z")
        ax[0].set_ylabel("y")
        ax[1].set_ylabel("z")
        ax[2].set_ylabel("y")

        plt.savefig(os.path.join(save_dir, f"{save_name_prefix}_batch{i_p3ds}_pred.pdf"))
        plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str)

    parser.add_argument(
        "--input_file", type=str, default=None, help="override config input_file"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="override config output_file"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="override config cache_dir"
    )
    parser.add_argument(
        "--plot_only", action="store_true", help="make some debuggin plots and exit"
    )

    args = parser.parse_args()

    overwrite_dict = {}
    if args.input_file is not None:
        overwrite_dict["input_file"] = args.input_file
    if args.output_file is not None:
        overwrite_dict["output_file"] = args.output_file
    if args.cache_dir is not None:
        overwrite_dict["cache_dir"] = args.cache_dir

    return args, overwrite_dict

if __name__ == "__main__":
    args, overwrite_dict = parse_arguments()
    main(args, overwrite_dict)
