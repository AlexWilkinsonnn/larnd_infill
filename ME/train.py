import time
from collections import defaultdict

import yaml
import numpy as np

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from ME.dataset import LarndDataset, DataPrepType, CollateCOO
from ME.models.completion_net import CompletionNet, CompletionNetSigMask
from aux import plot_ndlar_voxels_2

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)
DEVICE = torch.device("cuda:0")
DATA_PATH = "/share/rcifdata/awilkins/larnd_infill_data/zdownsample10/all"
VMAP_PATH = "/home/awilkins/larnd_infill/larnd_infill/voxel_maps/vmap_zdownresolution10.yml"

detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
geometry = get_geom_map(PIXEL_LAYOUT)

with open(VMAP_PATH, "r") as f:
    vmap = yaml.load(f, Loader=yaml.FullLoader)

net = CompletionNet(
    (vmap["n_voxels"]["x"], vmap["n_voxels"]["y"], vmap["n_voxels"]["z"]),
    in_nchannel=4, out_nchannel=1, final_pruning_threshold=(1 / 150)
)
net.to(DEVICE)

scalefactors = [1 / 150, 1 / 4]
dataset = LarndDataset(
    DATA_PATH, DataPrepType.GAP_DISTANCE, vmap, 2, 1, scalefactors, max_dataset_size=50000, seed=1
)

# collate_fn = CollateCOO(
#     coord_feat_pairs=(
#         ("input_coords", "input_feats"), ("target_coords", "target_feats"),
#         ("signal_mask_active_coords", "signal_mask_active_feats"),
#         ("signal_mask_gap_coords", "signal_mask_gap_feats")
#     )
# )
collate_fn = CollateCOO(
    coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
)

batch_size = 4
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=batch_size
)

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
print("LR {}".format(scheduler.get_lr()))

# crit = nn.BCEWithLogitsLoss()
crit = nn.MSELoss()
crit_zeromask = nn.MSELoss(reduction="sum")
crit_n_points = nn.L1Loss()
crit_n_points_zeromask = nn.L1Loss(reduction="sum")
# crit = nn.L1Loss()

net.train()
train_iter = iter(dataloader)

t0 = time.time()
losses_acc = defaultdict(list)
num_points_acc = []
for i in range(25000):
    data = next(train_iter)

    optimizer.zero_grad()

    s_in = ME.SparseTensor(
        coordinates=data["input_coords"], features=data["input_feats"], device=DEVICE
    )
    # s_sigmask_active = ME.SparseTensor(
    #     coordinates=data["signal_mask_active_coords"], features=data["signal_mask_active_feats"],
    #     device=DEVICE, coordinate_manager=s_in.coordinate_manager
    # )
    # s_sigmask_gap = ME.SparseTensor(
    #     coordinates=data["signal_mask_gap_coords"], features=data["signal_mask_gap_feats"],
    #     device=DEVICE, coordinate_manager=s_in.coordinate_manager
    # )
    # if s_sigmask_gap.F.shape[0]:
    #     s_in = s_in + s_sigmask_gap
    # print(s_sigmask_active.F.shape)
    # if s_sigmask_active.F.shape[0]:
    #     s_in = s_in + s_sigmask_active

    try:
        s_pred = net(s_in)
    except ValueError as e:
        print(s_in.C)
        print(s_in.C.shape)
        print(s_in.F.shape)
        raise e

    s_target = ME.SparseTensor(
        coordinates=data["target_coords"], features=data["target_feats"], device=DEVICE,
        coordinate_manager=s_in.coordinate_manager
    )

    all_coords = ME.MinkowskiUnion()(s_pred, s_target).C.type(torch.int)

    infill_mask = torch.zeros(all_coords.shape[0], dtype=torch.bool)
    x_mask, z_mask = data["mask_x"], data["mask_z"]
    for i_coord, coord in enumerate(all_coords):
        b_idx = coord[0].item()
        if coord[1].item() in x_mask[b_idx] or coord[3].item() in z_mask[b_idx]:
            infill_mask[i_coord] = True

    all_coords = all_coords.type(torch.float)

    infill_coords = all_coords[infill_mask]
    infill_coords_nonzero_mask = s_target.features_at_coordinates(infill_coords).squeeze() != 0
    infill_coords_nonzero = infill_coords[infill_coords_nonzero_mask]
    infill_coords_zero = infill_coords[~infill_coords_nonzero_mask]

    active_coords = all_coords[~infill_mask]
    active_coords_nonzero_mask = s_target.features_at_coordinates(active_coords).squeeze() != 0
    active_coords_nonzero = active_coords[active_coords_nonzero_mask]
    active_coords_zero = active_coords[~active_coords_nonzero_mask]

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
        print(s_pred.shape)
        print(infill_coords_zero.shape)
        print(s_pred.features_at_coordinates(infill_coords_zero))
        print(s_target.features_at_coordinates(infill_coords_zero))
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
    
    pred_active_features = (s_pred.features_at_coordinates(active_coords) != 0.0)
    target_active_features = (s_target.features_at_coordinates(active_coords) != 0.0)
    batch_active_coords = active_coords[:, 0] == 0
    if batch_active_coords.shape[0]:
        loss_active_n_points = crit_n_points(
            pred_active_features[batch_active_coords].sum(axis=0).type(torch.float),
            target_active_features[batch_active_coords].sum(axis=0).type(torch.float)
        )
    else:
        loss_active_n_points = crit_n_points_zeromask(
            pred_active_features[batch_active_coords].sum(axis=0).type(torch.float),
            target_active_features[batch_active_coords].sum(axis=0).type(torch.float)
        )
    for i_batch in range(1, batch_size):
        batch_active_coords = active_coords[:, 0] == i_batch
        if batch_active_coords.shape[0]:
            loss_active_n_points += crit_n_points(
                pred_active_features[batch_active_coords].sum(axis=0).type(torch.float),
                target_active_features[batch_active_coords].sum(axis=0).type(torch.float)
            )
        else:
            loss_active_n_points += crit_n_points_zeromask(
                pred_active_features[batch_active_coords].sum(axis=0).type(torch.float),
                target_active_features[batch_active_coords].sum(axis=0).type(torch.float)
            )
    loss_active_n_points = loss_active_n_points / batch_size

    pred_infill_features = (s_pred.features_at_coordinates(infill_coords) != 0.0)
    target_infill_features = (s_target.features_at_coordinates(infill_coords) != 0.0)
    batch_infill_coords = infill_coords[:, 0] == 0
    if batch_infill_coords.shape[0]:
        loss_infill_n_points = crit_n_points(
            pred_infill_features[batch_infill_coords].sum(axis=0).type(torch.float),
            target_infill_features[batch_infill_coords].sum(axis=0).type(torch.float)
        )
    else:
        loss_infill_n_points = crit_n_points_zeromask(
            pred_infill_features[batch_infill_coords].sum(axis=0).type(torch.float),
            target_infill_features[batch_infill_coords].sum(axis=0).type(torch.float)
        )
    for i_batch in range(1, batch_size):
        batch_infill_coords = infill_coords[:, 0] == i_batch
        if batch_infill_coords.shape[0]:
            loss_infill_n_points += crit_n_points(
                pred_infill_features[batch_infill_coords].sum(axis=0).type(torch.float),
                target_infill_features[batch_infill_coords].sum(axis=0).type(torch.float)
            )
        else:
            loss_infill_n_points += crit_n_points_zeromask(
                pred_infill_features[batch_infill_coords].sum(axis=0).type(torch.float),
                target_infill_features[batch_infill_coords].sum(axis=0).type(torch.float)
            )
    loss_active_n_points = loss_active_n_points / batch_size

    loss_tot = (
        loss_infill_zero + loss_infill_nonzero + 0.00001 * loss_infill_n_points +
        0.0001 * loss_active_zero +
        0.0001 * loss_active_nonzero +
        0.0001 * 0.00001 * loss_active_n_points
    )

    loss_tot.backward()
    losses_acc["tot"].append(loss_tot.item())
    losses_acc["infill_zero"].append(loss_infill_zero.item())
    losses_acc["infill_nonzero"].append(loss_infill_nonzero.item())
    losses_acc["infill_n_points"].append(loss_infill_n_points.item())
    losses_acc["active_zero"].append(loss_active_zero.item())
    losses_acc["active_nonzero"].append(loss_active_nonzero.item())
    losses_acc["active_n_points"].append(loss_active_n_points.item())
    optimizer.step()

    if (i + 1) % int(200 / batch_size) == 0:
        t_iter = time.time() - t0
        t0 = time.time()
        print(
            "Iter: {}, Time: {:.7f} ".format(i + 1, t_iter) +
            "Losses: total={:.7f} ".format(np.mean(losses_acc["tot"])) +
            "infill_zero={:.7f} ".format(np.mean(losses_acc["infill_zero"])) +
            "infill_nonzero={:.7f} ".format(np.mean(losses_acc["infill_nonzero"])) +
            "infill_n_points={:.7f} ".format(np.mean(losses_acc["infill_n_points"])) +
            "active_zero={:.7f} ".format(np.mean(losses_acc["active_zero"])) +
            "active_nonzero={:.7f}. ".format(np.mean(losses_acc["active_nonzero"])) +
            "active_n_points={:.7f}. ".format(np.mean(losses_acc["active_n_points"]))
        )
        losses = defaultdict(list)

    if (i + 1) % int(1000 / batch_size) == 0:
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

            n_voxels_active_pred = (feats_pred > 1).sum()
            print("Number of predicted coords: {}".format(feats_pred.size()))
            print("Number of predicted coords with adc > 1.0: {}".format(n_voxels_active_pred))

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
                saveas="tests/iter{}_batch{}_pred.pdf".format(i + 1, i_batch),
                max_feat=150
            )
            plot_ndlar_voxels_2(
                coords_packed_predonly, feats_predonly,
                detector,
                vmap["x"], vmap["y"], vmap["z"],
                batch_mask_x, batch_mask_z,
                saveas="tests/iter{}_batch{}_pred_predonly.pdf".format(i + 1, i_batch),
                max_feat=150
            )

    if (i + 1) % int(5000 / batch_size) == 0:
        scheduler.step()
        print("LR {}".format(scheduler.get_lr()))

""" Graveyard
    infill_coords = s_sigmask_gap.C.type(torch.float)
    infill_coords_zero_mask = s_target.features_at_coordinates(infill_coords).squeeze() == 0
    infill_coords_zero = infill_coords[infill_coords_zero_mask]
    infill_coords_nonzero = infill_coords[~infill_coords_zero_mask]

    active_coords = s_sigmask_active.C.type(torch.float)
    active_coords_zero_mask = s_target.features_at_coordinates(active_coords).squeeze() == 0
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
    # if infill_coords.shape[0]:
    #     loss_infill_sum = crit(
    #         s_pred.features_at_coordinates(infill_coords).squeeze().sum(),
    #         s_target.features_at_coordinates(infill_coords).squeeze().sum()
    #     )
    # else:
    #     loss_infill_sum = crit_zeromask(
    #         s_pred.features_at_coordinates(infill_coords).squeeze().sum(),
    #         s_target.features_at_coordinates(infill_coords).squeeze().sum()
    #     )

    loss_tot = (
        loss_infill_zero + loss_infill_nonzero +
        0.0001 * loss_active_zero + 0.0001 * loss_active_nonzero
    )

    loss_tot.backward()
    losses_acc["tot"].append(loss_tot.item())
    losses_acc["infill_zero"].append(loss_infill_zero.item())
    losses_acc["infill_nonzero"].append(loss_infill_nonzero.item())
    # losses_acc["infill_sum"].append(loss_infill_sum.item())
    losses_acc["active_zero"].append(loss_active_zero.item())
    losses_acc["active_nonzero"].append(loss_active_nonzero.item())
    optimizer.step()

    del infill_coords
    del infill_coords_zero_mask
    del loss_infill_zero
    del loss_infill_nonzero
    del active_coords
    del active_coords_zero_mask
    del loss_active_zero
    del loss_active_nonzero
    del loss_tot

    if (i + 1) % int(10 / batch_size) == 0:
        t_iter = time.time() - t0
        t0 = time.time()
        print(
            "Iter: {}, Time: {:.7f} ".format(i + 1, t_iter) +
            "Losses: total={:.7f} ".format(np.mean(losses_acc["tot"])) +
            "infill_zero={:.7f} ".format(np.mean(losses_acc["infill_zero"])) +
            "infill_nonzero={:.7f} ".format(np.mean(losses_acc["infill_nonzero"])) +
            # "infill_sum={:.7f} ".format(np.mean(losses_acc["infill_sum"])) +
            "active_zero={:.7f} ".format(np.mean(losses_acc["active_zero"])) +
            "active_nonzero={:.7f}".format(np.mean(losses_acc["active_nonzero"]))
        )
        losses = defaultdict(list)
"""
