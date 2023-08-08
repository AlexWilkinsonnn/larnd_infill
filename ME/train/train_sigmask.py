import time
from collections import defaultdict

import yaml
import numpy as np

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from ME.dataset import LarndDataset, DataPrepType, CollateCOO
from ME.models.completion_net import CompletionNetSigMask
from aux import plot_ndlar_voxels_2

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)
DEVICE = torch.device("cuda:0")
DATA_PATH = "/share/rcifdata/awilkins/larnd_infill_data/zdownsample10/all"
VMAP_PATH = "/home/awilkins/larnd_infill/larnd_infill/voxel_maps/vmap_zdownresolution10.yml"

def main():
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    # geometry = get_geom_map(PIXEL_LAYOUT)

    with open(VMAP_PATH, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)

    net = CompletionNetSigMask(
        (vmap["n_voxels"]["x"], vmap["n_voxels"]["y"], vmap["n_voxels"]["z"]),
        in_nchannel=3, out_nchannel=1, final_pruning_threshold=(1 / 150)
    )
    net.to(DEVICE)

    print(
        "Model has {:.1f} million parameters".format(
            sum(params.numel() for params in net.parameters()) / 1e6
        )
    )

    scalefactors = [1 / 150, 1 / 4]
    dataset = LarndDataset(
        DATA_PATH, DataPrepType.REFLECTION, vmap, 2, 1, scalefactors, max_dataset_size=5000, seed=1
    )

    collate_fn = CollateCOO(
        coord_feat_pairs=(
            ("input_coords", "input_feats"), ("target_coords", "target_feats"),
            ("signal_mask_active_coords", "signal_mask_active_feats"),
            ("signal_mask_gap_coords", "signal_mask_gap_feats")
        )
    )

    batch_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=batch_size
    )

    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    print("LR {}".format(scheduler.get_lr()))

    crit = nn.L1Loss()
    crit_zeromask = nn.L1Loss(reduction="sum")
    # crit_n_points = nn.L1Loss()
    # crit_n_points_zeromask = nn.L1Loss(reduction="sum")

    net.train()

    t0 = time.time()
    losses_acc = defaultdict(list)
    # num_points_acc = []
    epochs = 10
    n_iter = 0
    print_iter = 200
    plot_iter = 200
    lr_decay_iter = 5000

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))

        for data in dataloader:
            optimizer.zero_grad()

            try:
                s_in = ME.SparseTensor(
                    coordinates=data["input_coords"], features=data["input_feats"], device=DEVICE
                )
                s_sigmask_active = ME.SparseTensor(
                    coordinates=data["signal_mask_active_coords"],
                    features=data["signal_mask_active_feats"],
                    device=DEVICE, coordinate_manager=s_in.coordinate_manager
                )
                s_sigmask_gap = ME.SparseTensor(
                    coordinates=data["signal_mask_gap_coords"], features=data["signal_mask_gap_feats"],
                    device=DEVICE, coordinate_manager=s_in.coordinate_manager
                )
                s_target = ME.SparseTensor(
                    coordinates=data["target_coords"], features=data["target_feats"], device=DEVICE
                )
            except Exception as e:
                print("Failed to load into GPU sparse tensors - ", end="")
                print(e)
                continue

            if s_sigmask_gap.F.shape[0]:
                s_in = s_in + s_sigmask_gap
            if s_sigmask_active.F.shape[0]:
                s_in = s_in + s_sigmask_active

            try:
                s_pred = net(s_in)
            except Exception as e:
                # print(s_in.C)
                # print(s_in.C.shape)
                # print(s_in.F.shape)
                # print(s_sigmask_active.F.shape)
                # print(s_sigmask_gap.F.shape)
                print("net(s_in) failed - ", end="")
                print(e)
                continue

            ret = calc_losses(
                s_pred, s_target, s_sigmask_gap, s_sigmask_active, crit, crit_zeromask
            )
            loss_infill_zero, loss_infill_nonzero, loss_active_zero, loss_active_nonzero = ret

            loss_tot = (
                0.0000001 * loss_infill_zero + loss_infill_nonzero +
                0.0001 * loss_active_zero + 0.0001 * loss_active_nonzero
            )

            loss_tot.backward()
            losses_acc["tot"].append(loss_tot.item())
            losses_acc["infill_zero"].append(loss_infill_zero.item())
            losses_acc["infill_nonzero"].append(loss_infill_nonzero.item())
            losses_acc["active_zero"].append(loss_active_zero.item())
            losses_acc["active_nonzero"].append(loss_active_nonzero.item())

            optimizer.step()

            if (n_iter + 1) % int(print_iter / batch_size) == 0:
                t_iter = time.time() - t0
                t0 = time.time()
                print_losses(losses_acc, n_iter, t_iter, s_pred)
                losses_acc = defaultdict(list)

            if (n_iter + 1) % int(plot_iter / batch_size) == 0:
                plot_pred(s_pred, s_target, data, vmap, scalefactors, n_iter, detector)

            if (n_iter + 1) % int(lr_decay_iter / batch_size) == 0:
                scheduler.step()
                print("LR {}".format(scheduler.get_lr()))

            n_iter += 1


def calc_losses(s_pred, s_target, s_sigmask_gap, s_sigmask_active, crit, crit_zeromask):
    infill_coords = s_sigmask_gap.C.type(torch.float)
    infill_coords_zero_mask = s_target.features_at_coordinates(infill_coords)[:, 0] == 0
    infill_coords_zero = infill_coords[infill_coords_zero_mask]
    infill_coords_nonzero = infill_coords[~infill_coords_zero_mask]

    active_coords = s_sigmask_active.C.type(torch.float)
    active_coords_zero_mask = s_target.features_at_coordinates(active_coords)[:, 0] == 0
    active_coords_zero = active_coords[active_coords_zero_mask]
    active_coords_nonzero = active_coords[~active_coords_zero_mask]

    try:
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
    except Exception as e:
        print("s_pred.shape: {}".format(s_pred.shape))
        print("s_sigmask_gap.shape: {}".format(s_sigmask_gap.shape))
        print("s_sigmask_active.shape: {}".format(s_sigmask_active.shape))
        raise e

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


def plot_pred(s_pred, s_target, data, vmap, scalefactors, n_iter, detector):
    coords_packed = [[], [], []]
    feats = []
    for i_batch, (coords_pred, feats_pred, coords_target, feats_target) in enumerate(
        zip(
            *s_pred.decomposed_coordinates_and_features,
            *s_target.decomposed_coordinates_and_features
        )
    ):
        if i_batch > 2:
            break

        coords_pred, feats_pred = coords_pred.cpu(), feats_pred.cpu() * (1 / scalefactors[0])
        coords_target, feats_target = (
            coords_target.cpu(), feats_target.cpu() * (1 / scalefactors[0])
        )
        batch_mask_x, batch_mask_z = data["mask_x"][i_batch], data["mask_z"][i_batch]

        coords_packed, feats = [[], [], []], []
        coords_packed_predonly, feats_predonly = [[], [], []], []

        for coord, feat in zip(coords_pred, feats_pred):
            if feat.item() < 1:
                continue
            if coord[0].item() in batch_mask_x or coord[2].item() in batch_mask_z:
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats.append(feat.item())
            coords_packed_predonly[0].append(coord[0].item())
            coords_packed_predonly[1].append(coord[1].item())
            coords_packed_predonly[2].append(coord[2].item())
            feats_predonly.append(feat.item())
        for coord, feat in zip(coords_target, feats_target):
            if coord[0].item() not in batch_mask_x and coord[2].item() not in batch_mask_z:
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats.append(feat.item())

        plot_ndlar_voxels_2(
            coords_packed, feats,
            detector,
            vmap["x"], vmap["y"], vmap["z"],
            batch_mask_x, batch_mask_z,
            saveas="tests/iter{}_batch{}_pred.pdf".format(n_iter + 1, i_batch),
            max_feat=150
        )
        plot_ndlar_voxels_2(
            coords_packed_predonly, feats_predonly,
            detector,
            vmap["x"], vmap["y"], vmap["z"],
            batch_mask_x, batch_mask_z,
            saveas="tests/iter{}_batch{}_pred_predonly.pdf".format(n_iter + 1, i_batch),
            max_feat=150
        )


if __name__ == "__main__":
    main()

