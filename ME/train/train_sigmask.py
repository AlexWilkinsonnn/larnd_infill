import time, argparse, os
from collections import defaultdict

import yaml
import numpy as np

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties

from ME.config_parser import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net import CompletionNetSigMask
from aux import plot_ndlar_voxels_2

# DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
# PIXEL_LAYOUT=(
#     "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
# )
# DEVICE = torch.device("cuda:0")
# DATA_PATH = "/share/rcifdata/awilkins/larnd_infill_data/zdownsample10/all"
# VMAP_PATH = "/home/awilkins/larnd_infill/larnd_infill/voxel_maps/vmap_zdownresolution10.yml"

def main(args):
    conf = get_config(args.config)

    device = torch.device(conf.device)

    net = CompletionNetSigMask(
        (conf.vmap["n_voxels"]["x"], conf.vmap["n_voxels"]["y"], conf.vmap["n_voxels"]["z"]),
        in_nchannel=conf.n_feats_in + 1, out_nchannel=conf.n_feats_out,
        final_pruning_threshold=conf.scalefactors[0]
    )
    net.to(device)

    print(
        "Model has {:.1f} million parameters".format(
            sum(params.numel() for params in net.parameters()) / 1e6
        )
    )

    dataset = LarndDataset(
        conf.data_path,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        max_dataset_size=conf.max_dataset_size
    )

    collate_fn = CollateCOO(
        coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=max(conf.max_num_workers, conf.batch_size),
        shuffle=True
    )

    optimizer = optim.SGD(net.parameters(), lr=conf.initial_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    print("LR {}".format(scheduler.get_lr()))

    crit, crit_zeromask = init_loss_func(conf.loss_func)

    net.train()

    t0 = time.time()
    losses_acc = defaultdict(list)
    n_iter = 0

    for epoch in range(conf.epochs):
        print("Epoch {}".format(epoch))

        for data in dataloader:
            optimizer.zero_grad()

            try:
                s_in = ME.SparseTensor(
                    coordinates=data["input_coords"], features=data["input_feats"], device=device
                )
                s_target = ME.SparseTensor(
                    coordinates=data["target_coords"], features=data["target_feats"], device=device
                )
            except Exception as e:
                print("Failed to load in GPU sparse tensors - ", end="")
                print(e)
                continue

            s_pred = net(s_in)

            ret = calc_losses(s_pred, s_in, s_target, crit, crit_zeromask)
            loss_infill_zero, loss_infill_nonzero, loss_active_zero, loss_active_nonzero = ret

            loss_tot = (
                conf.loss_infill_zero_weight * loss_infill_zero +
                conf.loss_infill_nonzero_weight * loss_infill_nonzero +
                conf.loss_active_zero_weight * loss_active_zero +
                conf.loss_active_nonzero_weight * loss_active_nonzero
            )

            loss_tot.backward()
            losses_acc["tot"].append(loss_tot.item())
            losses_acc["infill_zero"].append(loss_infill_zero.item())
            losses_acc["infill_nonzero"].append(loss_infill_nonzero.item())
            losses_acc["active_zero"].append(loss_active_zero.item())
            losses_acc["active_nonzero"].append(loss_active_nonzero.item())

            optimizer.step()

            if (n_iter + 1) % int(args.print_iter / conf.batch_size) == 0:
                t_iter = time.time() - t0
                t0 = time.time()
                print_losses(losses_acc, n_iter, t_iter, s_pred)
                losses_acc = defaultdict(list)

            if (n_iter + 1) % int(args.valid_iter / conf.batch_size) == 0:
                plot_pred(
                    s_pred, s_in, s_target,
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    n_iter,
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoints_dir, conf.name),
                    save_tensors=True
                )
            elif (n_iter + 1) % int(args.plot_iter / conf.batch_size) == 0:
                plot_pred(
                    s_pred, s_in, s_target,
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    n_iter,
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoints_dir, conf.name)
                )

            if (n_iter + 1) % int(conf.lr_decay_iter / conf.batch_size) == 0:
                scheduler.step()
                print("LR {}".format(scheduler.get_lr()))

            n_iter += 1

    plot_pred(
        s_pred, s_in, s_target, data, conf.vmap, conf.scalefactors, n_iter, conf.detector,
        save_dir=os.path.join(conf.checkpoints_dir, conf.name), save_tensors=True
    )

def init_loss_func(loss_func):
    if loss_func == "L1Loss":
        crit = nn.L1Loss()
        crit_zeromask = nn.L1Loss(reduction="sum")
    elif loss_func == "MSELoss":
        crit = nn.MSELoss()
        crit_zeromask = nn.MSELoss(reduction="sum")
    else:
        raise NotImplementedError("loss_func={} not valid".format(loss_func))

    return crit, crit_zeromask

def calc_losses(s_pred, s_in, s_target, crit, crit_zeromask):
    s_in_infill_mask = s_in.F[:, -1] == 1
    infill_coords = s_in.C[s_in_infill_mask].type(torch.float)
    active_coords = s_in.C[~s_in_infill_mask].type(torch.float)

    infill_coords_zero_mask = s_target.features_at_coordinates(infill_coords)[:, 0] == 0
    infill_coords_zero = infill_coords[infill_coords_zero_mask]
    infill_coords_nonzero = infill_coords[~infill_coords_zero_mask]

    active_coords_zero_mask = s_target.features_at_coordinates(active_coords)[:, 0] == 0
    active_coords_zero = active_coords[active_coords_zero_mask]
    active_coords_nonzero = active_coords[~active_coords_zero_mask]

    if infill_coords_zero.shape[0]:
        loss_infill_zero = crit(
            s_pred.features_at_coordinates(infill_coords_zero).squeeze(),
            s_target.features_at_coordinates(infill_coords_zero).squeeze()
        )
    else:
        loss_infill_zero = crit_zeromask(
            s_pred.features_at_coordinates(infill_coords_zero).squeeze(),
            s_target.features_at_coordinates(infill_coords_zero).squeeze()
        )

    if infill_coords_nonzero.shape[0]:
        loss_infill_nonzero = crit(
            s_pred.features_at_coordinates(infill_coords_nonzero).squeeze(),
            s_target.features_at_coordinates(infill_coords_nonzero).squeeze()
        )
    else:
        loss_infill_nonzero = crit_zeromask(
            s_pred.features_at_coordinates(infill_coords_nonzero).squeeze(),
            s_target.features_at_coordinates(infill_coords_nonzero).squeeze()
        )
    if active_coords_zero.shape[0]:
        loss_active_zero = crit(
            s_pred.features_at_coordinates(active_coords_zero).squeeze(),
            s_target.features_at_coordinates(active_coords_zero).squeeze()
        )
    else:
        loss_active_zero = crit_zeromask(
            s_pred.features_at_coordinates(active_coords_zero).squeeze(),
            s_target.features_at_coordinates(active_coords_zero).squeeze()
        )
    if active_coords_nonzero.shape[0]:
        loss_active_nonzero = crit(
            s_pred.features_at_coordinates(active_coords_nonzero).squeeze(),
            s_target.features_at_coordinates(active_coords_nonzero).squeeze()
        )
    else:
        loss_active_nonzero = crit_zeromask(
            s_pred.features_at_coordinates(active_coords_nonzero).squeeze(),
            s_target.features_at_coordinates(active_coords_nonzero).squeeze()
        )

    return loss_infill_zero, loss_infill_nonzero, loss_active_zero, loss_active_nonzero


def print_losses(losses_acc, n_iter, t_iter, s_pred):
    print(
        "Iter: {}, Time: {:.7f}, last s_pred.shape: {} ".format(n_iter + 1, t_iter, s_pred.shape) +
        "Losses: total={:.7f} ".format(np.mean(losses_acc["tot"])) +
        "infill_zero={:.7f} ".format(np.mean(losses_acc["infill_zero"])) +
        "infill_nonzero={:.7f} ".format(np.mean(losses_acc["infill_nonzero"])) +
        "active_zero={:.7f} ".format(np.mean(losses_acc["active_zero"])) +
        "active_nonzero={:.7f} ".format(np.mean(losses_acc["active_nonzero"]))
    )


def plot_pred(
    s_pred, s_in, s_target, data, vmap, scalefactors, n_iter, detector,
    max_evs=2, save_dir="test/", save_tensors=False
):
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

        coords_pred, feats_pred = coords_pred.cpu(), feats_pred.cpu() * (1 / scalefactors[0])
        coords_target, feats_target = (
            coords_target.cpu(), feats_target.cpu() * (1 / scalefactors[0])
        )
        coords_in, feats_in = coords_in.cpu(), feats_in.cpu()
        batch_mask_x, batch_mask_z = data["mask_x"][i_batch], data["mask_z"][i_batch]

        coords_packed, feats_list = [[], [], []], []
        coords_packed_predonly, feats_list_predonly = [[], [], []], []
        coords_sigmask_gap_packed = [[], [], []]
        coords_sigmask_active_packed = [[], [], []]
        coords_target_packed, feats_list_target = [[], [], []], []

        for coord, feat in zip(coords_pred, feats_pred):
            if coord[0].item() in batch_mask_x or coord[2].item() in batch_mask_z:
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats_list.append(feat.item())
            coords_packed_predonly[0].append(coord[0].item())
            coords_packed_predonly[1].append(coord[1].item())
            coords_packed_predonly[2].append(coord[2].item())
            feats_list_predonly.append(feat.item())
        for coord, feat in zip(coords_target, feats_target):
            coords_target_packed[0].append(coord[0].item())
            coords_target_packed[1].append(coord[1].item())
            coords_target_packed[2].append(coord[2].item())
            feats_list_target.append(feat.item())
            if coord[0].item() not in batch_mask_x and coord[2].item() not in batch_mask_z:
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats_list.append(feat.item())

        for coord, feat in zip(coords_in, feats_in):
            if feat[-1]:
                coords_sigmask_gap_packed[0].append(coord[0].item())
                coords_sigmask_gap_packed[1].append(coord[1].item())
                coords_sigmask_gap_packed[2].append(coord[2].item())
            else:
                coords_sigmask_active_packed[0].append(coord[0].item())
                coords_sigmask_active_packed[1].append(coord[1].item())
                coords_sigmask_active_packed[2].append(coord[2].item())

        plot_ndlar_voxels_2(
            coords_packed, feats_list,
            detector,
            vmap["x"], vmap["y"], vmap["z"],
            batch_mask_x, batch_mask_z,
            saveas=os.path.join(save_dir, "iter{}_batch{}_pred.pdf".format(n_iter + 1, i_batch)),
            max_feat=150,
            signal_mask_gap_coords=coords_sigmask_gap_packed,
            signal_mask_active_coords=coords_sigmask_active_packed
        )
        plot_ndlar_voxels_2(
            coords_packed_predonly, feats_list_predonly,
            detector,
            vmap["x"], vmap["y"], vmap["z"],
            batch_mask_x, batch_mask_z,
            saveas=os.path.join(
                save_dir, "iter{}_batch{}_pred_predonly.pdf".format(n_iter + 1, i_batch)
            ),
            max_feat=150,
            signal_mask_gap_coords=coords_sigmask_gap_packed,
            signal_mask_active_coords=coords_sigmask_active_packed
        )
        plot_ndlar_voxels_2(
            coords_target_packed, feats_list_target,
            detector,
            vmap["x"], vmap["y"], vmap["z"],
            batch_mask_x, batch_mask_z,
            saveas=os.path.join(save_dir, "iter{}_batch{}_target.pdf".format(n_iter + 1, i_batch)),
            max_feat=150,
            signal_mask_gap_coords=coords_sigmask_gap_packed,
            signal_mask_active_coords=coords_sigmask_active_packed
        )

        if save_tensors:
            coords_in_packed, feats_in_list = [[], [], []], []
            for coord, feat in zip(coords_in, feats_in):
                coords_in_packed[0].append(coord[0].item())
                coords_in_packed[1].append(coord[1].item())
                coords_in_packed[2].append(coord[2].item())
                feats_in_list.append(feat.tolist())
            
            in_dict = {
                tuple(coord.tolist()) : feat.tolist() for coord, feat in zip(coords_in, feats_in)
            }
            with open(
                os.path.join(save_dir,"iter{}_batch{}_in.yml".format(n_iter + 1, i_batch)), "w"
            ) as f:
                yaml.dump(in_dict, f)

            pred_dict = {
                tuple(coord.tolist()) : feat.tolist()
                for coord, feat in zip(coords_pred, feats_pred)
            }
            with open(
                os.path.join(save_dir,"iter{}_batch{}_pred.yml".format(n_iter + 1, i_batch)), "w"
            ) as f:
                yaml.dump(pred_dict, f)

            target_dict = {
                tuple(coord.tolist()) : feat.tolist()
                for coord, feat in zip(coords_target, feats_target)
            }
            with open(
                os.path.join(save_dir,"iter{}_batch{}_target.yml".format(n_iter + 1, i_batch)), "w"
            ) as f:
                yaml.dump(target_dict, f)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument("--plot_iter", type=int, default=1000)
    parser.add_argument("--valid_iter", type=int, default=5000)
    parser.add_argument("--print_iter", type=int, default=200)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

