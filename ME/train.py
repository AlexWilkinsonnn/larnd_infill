import time

import yaml
import numpy as np

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from ME.dataset import LarndDataset, MaskType, CollateCOO
from ME.models.completion_net import CompletionNet
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
    in_nchannel=2, out_nchannel=1
)
net.to(DEVICE)

dataset = LarndDataset(DATA_PATH, MaskType.LOSS_ONLY, vmap, max_dataset_size=50000, seed=1)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=CollateCOO(DEVICE))

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
print("LR {}".format(scheduler.get_lr()))

# crit = nn.BCEWithLogitsLoss()
crit = nn.MSELoss()

net.train()
train_iter = iter(dataloader)

t0 = time.time()
loss_acc = []
for i in range(500000):
    data = next(train_iter)

    optimizer.zero_grad()

    # s_in, in_coords, in_feats = data["input"], data["input_coords"], data["input_feats"]
    # target_coords, target_feats = data["target_coords"], data["target_feats"]
    s_in, s_target = data["input"], data["target"]
    mask_x, mask_z = data["mask_x"], data["mask_z"]

    s_pred = net(s_in)
    target_feats_padded = s_target.features_at_coordinates(s_pred.C.type(torch.float32))
    loss = crit(s_pred.F.squeeze(), target_feats_padded.squeeze())

    loss.backward()
    loss_acc.append(loss.item())
    optimizer.step()
    t2 = time.time() - t0

    if (i + 1) % 200 == 0:
        t_iter = time.time() - t0
        t0 = time.time()
        print("Iter: {}, Loss: {:.3f}, Time: {:.3f}".format(i + 1, np.mean(loss_acc), t_iter))
        loss_acc = []

    if (i + 1) % 1000 == 0:
        coords_packed = [[], [], []]
        feats = []
        for i_batch, (coords_pred, feats_pred, coords_target, feats_target) in enumerate(
            zip(
                *s_pred.decomposed_coordinates_and_features,
                *s_target.decomposed_coordinates_and_features
            )
        ):
            coords_pred = coords_pred.cpu()
            feats_pred = feats_pred.cpu()
            coords_target = coords_target.cpu()
            feats_target = feats_target.cpu()
            batch_mask_x = mask_x[i_batch]
            batch_mask_z = mask_z[i_batch]
            coords_packed = [[], [], []]
            feats = []
            for coord, feat in zip(coords_pred, feats_pred):
                if coord[0].item() in batch_mask_x or coord[2].item() in batch_mask_z:
                    coords_packed[0].append(coord[0].item())
                    coords_packed[1].append(coord[1].item())
                    coords_packed[2].append(coord[2].item())
                    feats.append(feat.item())
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
                saveas="tests/iter{}_pred.pdf".format(i + 1)
            )

    if (i + 1) % 5000 == 0:
        scheduler.step()
        print("LR {}".format(scheduler.get_lr()))

