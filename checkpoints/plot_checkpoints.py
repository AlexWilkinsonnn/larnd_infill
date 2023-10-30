import argparse, os, glob

import yaml

from ME.config_parser import get_config
from aux import plot_ndlar_voxels_2


def main(args):
    overwrite_dict = {}
    if args.det_props:
        overwrite_dict["det_props"] = args.det_props
    if args.pixel_layout:
        overwrite_dict["pixel_layout"] = args.pixel_layout
    if args.vmap_path:
        overwrite_dict["vmap_path"] = args.vmap_path

    conf = get_config(
        os.path.join(args.checkpoint_dir, os.path.basename(args.checkpoint_dir) + ".yml"),
        overwrite_dict=overwrite_dict, prep_checkpoint_dir=False
    )
    
    with open(
        os.path.join(args.checkpoint_dir, "iter{}_batch0_in.yml".format(args.iter))
    ) as f:
        in_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(args.checkpoint_dir, "iter{}_batch0_pred.yml".format(args.iter))
    ) as f:
        pred_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(
        os.path.join(args.checkpoint_dir, "iter{}_batch0_target.yml".format(args.iter))
    ) as f:
        target_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    coords_packed_out, feats_out = [ [], [], [] ], []
    coords_packed_signal_mask_active =  [ [], [], [] ]
    coords_packed_signal_mask_gap = [ [], [], [] ]
    mask_x, mask_z = set(), set()
    coords_set_signal_mask_active = set()

    for coord, feat in in_dict.items():
        if feat[0]:
            coords_packed_out[0].append(coord[0])
            coords_packed_out[1].append(coord[1])
            coords_packed_out[2].append(coord[2])
            feats_out.append(feat[0] * 150)
        # elif feat[-1]:
        #     coords_packed_signal_mask_gap[0].append(coord[0])
        #     coords_packed_signal_mask_gap[1].append(coord[1])
        #     coords_packed_signal_mask_gap[2].append(coord[2])
        #     # mask_x.add(coord[0])
        #     # mask_z.add(coord[2])
        # else:
        #     coords_packed_signal_mask_active[0].append(coord[0])
        #     coords_packed_signal_mask_active[1].append(coord[1])
        #     coords_packed_signal_mask_active[2].append(coord[2])
        #     coords_set_signal_mask_active.add((coord[0], coord[1], coord[2]))

    # for coord, feat in pred_dict.items():
    for coord, feat in target_dict.items():
        if (
            feat[-1] and feat[0] > 4 and
            (coord[0], coord[1], coord[2]) not in coords_set_signal_mask_active # NOTE not sure why I need this
        ):
            coords_packed_out[0].append(coord[0])
            coords_packed_out[1].append(coord[1])
            coords_packed_out[2].append(coord[2])
            feats_out.append(feat[0])

    mask_x, mask_z = set(conf.vmap["x_gaps"]), set(conf.vmap["z_gaps"])

    plot_ndlar_voxels_2(
        coords_packed_out, feats_out,
        conf.detector,
        conf.vmap["x"], conf.vmap["y"], conf.vmap["z"],
        mask_x, mask_z,
        max_feat=150, signal_mask_gap_coords=coords_packed_signal_mask_gap,
        signal_mask_active_coords=coords_packed_signal_mask_active,
        z_scalefactor=6
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint_dir")
    parser.add_argument("iter", type=int)

    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--det_props", default="", type=str)
    parser.add_argument("--pixel_layout", default="", type=str)
    parser.add_argument("--vmap_path", default="", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
