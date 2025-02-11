# NOTE Using this to make plots for my thesis by calling it like:
# python plots.py -n 10 {--reflections,--mask,...} experiments/nu_infillcuts_forwardbackwardzshift038_zdownsample10/exp3_1_plots.yml

import argparse, os

from tqdm import tqdm
from matplotlib import pyplot as plt; import matplotlib

import torch;
import MinkowskiEngine as ME

from ME.config_parsers.parser_train import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net_adversarial import CompletionNetAdversarial

# For mask plots
# FIGSIZE=(8, 4.5)
# FONTSIZE=13
# LABELSIZE=10
# For reflection plots
FIGSIZE=(6, 5)
FONTSIZE=13
LABELSIZE=10

def main(args):
    conf = get_config(args.config)

    if not os.path.exists(os.path.join(conf.checkpoint_dir, "thesis_plots")):
        os.makedirs(os.path.join(conf.checkpoint_dir, "thesis_plots"))

    global g_x_true_gaps
    g_x_true_gaps = conf.vmap["x_gaps"]
    global g_z_true_gaps
    g_z_true_gaps = conf.vmap["z_gaps"]

    model = CompletionNetAdversarial(conf)
    model.eval()

    collate_fn = CollateCOO(
        coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    )
    dataset = LarndDataset(
        conf.test_data_path if args.use_test_data else conf.valid_data_path,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        conf.xyz_smear_infill, conf.xyz_smear_active,
        conf.xyz_max_reflect_distance,
        max_dataset_size=-1 if args.use_test_data else conf.max_valid_dataset_size,
        seed=1
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False
    )
    if args.use_true_gaps:
        dataloader.dataset.set_use_true_gaps(True)

    if args.mask or args.reflections or args.preds:
        for i_data, data in tqdm(enumerate(dataloader), desc="Val Loop"):
            if i_data < args.n_skip:
                continue
            if i_data >= args.n:
                break

            model.set_input(data)
            model.test(compute_losses=True)

            vis = model.get_current_visuals()

            if args.mask:
                plot_a_thing(
                    vis["s_pred"], vis["s_in"], vis["s_target"],
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    "iter{}-valid".format(i_data), "masking",
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoint_dir, "thesis_plots"),
                    show_mask=True
                )

            if args.reflections:
                plot_a_thing(
                    vis["s_pred"], vis["s_in"], vis["s_target"],
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    "iter{}-valid".format(i_data), "reflections",
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoint_dir, "thesis_plots"),
                    show_reflections=True
                )

            if args.preds:
                plot_a_thing(
                    vis["s_pred"], vis["s_in"], vis["s_target"],
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    "iter{}-valid".format(i_data), "pred",
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoint_dir, "thesis_plots"),
                    show_preds=True
                )


def plot_a_thing(
    s_pred, s_in, s_target,
    data,
    vmap,
    scalefactors,
    save_name_prefix, save_name_suffix,
    detector,
    max_evs=6,
    save_dir="test/",
    z_scalefactor=1,
    show_mask=False, show_reflections=False, show_preds=False
):
    x_vmap, z_vmap = vmap["x"], vmap["z"]

    for i_batch, (
        coords_pred, feats_pred, coords_target, feats_target, coords_in, feats_in
    ) in enumerate(
        zip(
            *s_pred.decomposed_coordinates_and_features,
            *s_target.decomposed_coordinates_and_features,
            *s_in.decomposed_coordinates_and_features
        )
    ):
        if i_batch >= max_evs:
            break

        # Get coordinates and features
        coords_target, feats_target = (
            coords_target.cpu(), feats_target.cpu() * (1 / scalefactors[0])
        )
        coords_in, feats_in = coords_in.cpu(), feats_in.cpu()
        x_gaps, z_gaps = data["mask_x"][i_batch], data["mask_z"][i_batch]

        coords_packed, feats_list = [[], [], []], []
        for coord, feat in zip(coords_target, feats_target):
            if show_preds and (coord[0].item() in x_gaps or coord[2].item() in z_gaps):
                continue
            coords_packed[0].append(coord[0].item())
            coords_packed[1].append(coord[1].item())
            coords_packed[2].append(coord[2].item())
            feats_list.append(int(feat.item()))
        if show_preds:
            coords_pred, feats_pred = (
                coords_pred.cpu(), feats_pred.cpu() * (1 / scalefactors[0])
            )
            for coord, feat in zip(coords_pred, feats_pred):
                if coord[0].item() not in x_gaps and coord[2].item() not in z_gaps:
                    continue
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats_list.append(int(feat.item()))

        coords_sigmask_gap_packed = [[], [], []]
        for coord, feat in zip(coords_in, feats_in):
            if feat[-1]:
                coords_sigmask_gap_packed[0].append(coord[0].item())
                coords_sigmask_gap_packed[1].append(coord[1].item())
                coords_sigmask_gap_packed[2].append(coord[2].item())

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

        norm_feats = matplotlib.colors.Normalize(vmin=0, vmax=max(feats_list))
        m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.cividis)

        # Draw shaded regions for mask and inactive volumes
        for x_gap_coord in x_gaps:
            x_bin = x_vmap[x_gap_coord]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            z_pos = detector.tpc_borders[-1][2][0]
            z_size = detector.tpc_borders[0][2][0] - detector.tpc_borders[-1][2][0]
            ax.add_patch(
                matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="gray", alpha=0.3)
            )
        if show_mask:
            for x_gap_coord in g_x_true_gaps:
                x_bin = x_vmap[x_gap_coord]
                x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
                z_pos = detector.tpc_borders[-1][2][0]
                z_size = detector.tpc_borders[0][2][0] - detector.tpc_borders[-1][2][0]
                ax.add_patch(
                    matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="blue", alpha=0.1)
                )
        for z_gap_coord in z_gaps:
            z_bin = z_vmap[z_gap_coord]
            z_size, z_pos = z_bin[1] - z_bin[0], z_bin[0]
            x_pos = detector.tpc_borders[-1][0][1]
            x_size = detector.tpc_borders[0][0][0] - detector.tpc_borders[-1][0][1]
            ax.add_patch(
                matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="gray", alpha=0.3)
            )
        if show_mask:
            for z_gap_coord in g_z_true_gaps:
                z_bin = z_vmap[z_gap_coord]
                z_size, z_pos = z_bin[1] - z_bin[0], z_bin[0]
                x_pos = detector.tpc_borders[-1][0][1]
                x_size = detector.tpc_borders[0][0][0] - detector.tpc_borders[-1][0][1]
                ax.add_patch(
                    matplotlib.patches.Rectangle((x_pos, z_pos), x_size, z_size, fc="blue", alpha=0.1)
                )

        # Draw packets
        curr_patches_xz = set()
        for coord_x, coord_y, coord_z, feat in zip(*coords_packed, feats_list):
            x_bin = x_vmap[coord_x]
            x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
            z_bin = z_vmap[coord_z]
            z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

            c = m_feats.to_rgba(feat)
            alpha = 1.0

            pos_xz = (x_pos, z_pos)
            if pos_xz not in curr_patches_xz:
                curr_patches_xz.add(pos_xz)
                ax.add_patch(
                    matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
                )

        # Draw candidate infill coords from reflections
        if show_reflections:
            curr_patches_reflections_xz = set()
            for coord_x, coord_y, coord_z in zip(*coords_sigmask_gap_packed):
                x_bin = x_vmap[coord_x]
                x_size, x_pos = x_bin[1] - x_bin[0], x_bin[0]
                z_bin = z_vmap[coord_z]
                z_size, z_pos = (z_bin[1] - z_bin[0]) * z_scalefactor, z_bin[0]

                c = "green"
                alpha = 0.3

                pos_xz = (x_pos, z_pos)
                if pos_xz not in curr_patches_xz and pos_xz not in curr_patches_reflections_xz:
                    curr_patches_reflections_xz.add(pos_xz)
                    ax.add_patch(
                        matplotlib.patches.Rectangle(pos_xz, x_size, z_size, fc=c, alpha=alpha)
                    )

        # Styling plot...
        max_x = detector.tpc_borders[0][0][0]
        min_x = detector.tpc_borders[-1][0][1]
        max_z = detector.tpc_borders[0][2][0]
        min_z = detector.tpc_borders[-1][2][0]
        # XXX Edit these manually
        # For mask, iter5 batch5
        # min_x, max_x = 580, 730
        # min_z, max_z = -60, 100
        # For reflections, iter8 batch3
        # min_x, max_x = 492, 545
        # min_z, max_z = -50, -135
        # For reflections, iter0 batch5
        # min_x, max_x = 530, 610
        # min_z, max_z = -55, -115
        # For reflections, iter10 batch5
        # min_x, max_x = 630, 780
        # min_z, max_z = 130, 180
        # For dummy_edep_fixz_fixy pred true gaps, iter0 batch2
        # min_x, max_x = 590, 640
        # min_z, max_z = 260, 290
        # For dummy_edep_fixz pred true gaps, iter0 batch4
        # min_x, max_x = 595, 645
        # min_z, max_z = 190, 230
        # For dummy_edep pred true gaps, iter0 batch0
        # min_x, max_x = 790, 840
        # min_z, max_z = 20, 70
        # For gps_single_muon pred true gaps, iter0 batch2
        # min_x, max_x = 650, 750
        # min_z, max_z = 0, 100
        # For gps_single_muon pred true gaps, iter4 batch3
        # min_x, max_x = 485, 550
        # min_z, max_z = -150, -85
        # For gps_multi_muon pred true gaps, iter16 batch1
        # min_x, max_x = 470, 550
        # min_z, max_z = 80, 180

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_z, max_z)
        ax.set_xlabel("x (cm)", fontsize=FONTSIZE)
        ax.set_ylabel("z (cm)", fontsize=FONTSIZE)
        ax.xaxis.set_tick_params(labelsize=LABELSIZE)
        ax.yaxis.set_tick_params(labelsize=LABELSIZE)

        plt.savefig(
            os.path.join(save_dir, f"{save_name_prefix}_batch{i_batch}_{save_name_suffix}.pdf"),
            bbox_inches="tight"
        )
        plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--n_skip", type=int, default=0)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--reflections", action="store_true")
    parser.add_argument("--preds", action="store_true")
    parser.add_argument("--use_true_gaps", action="store_true")
    parser.add_argument("--use_test_data", action="store_true")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
