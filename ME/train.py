import time

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from ME.dataset import LarndDataset, MaskType, CollateCOO
from ME.models.completion_net import CompletionNet
from aux import plot_ndlar_voxels

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)

DEVICE = torch.device("cuda:0")

PATH = "/share/rcifdata/awilkins/larnd_infill_data/all"
PIXEL_COLS_PER_ANODE = 256
PIXEL_COLS_PER_GAP = 11 # 4.14 / 0.38
TICKS_PER_MODULE = 6117
TICKS_PER_GAP = 79 # 1.3cm / (0.1us * 0.1648cm/us)
X_GAPS = []
for i in range(5):
    X_GAPS.append(PIXEL_COLS_PER_ANODE * (i + 1) + PIXEL_COLS_PER_GAP * i)
    X_GAPS.append(PIXEL_COLS_PER_ANODE * (i + 1) + PIXEL_COLS_PER_GAP * (i + 1) - 1)
Z_GAPS = []
for i in range(7):
    Z_GAPS.append(TICKS_PER_MODULE * (i + 1) + TICKS_PER_GAP * i)
    Z_GAPS.append(TICKS_PER_MODULE * (i + 1) + TICKS_PER_GAP * (i + 1) - 1)

detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
geometry = get_geom_map(PIXEL_LAYOUT)

net = CompletionNet(in_nchannel=1, out_nchannel=1).to(DEVICE)
dataset = LarndDataset(
    PATH, MaskType.LOSS_ONLY,
    X_GAPS, PIXEL_COLS_PER_ANODE, PIXEL_COLS_PER_GAP, PIXEL_COLS_PER_GAP,
    Z_GAPS, TICKS_PER_MODULE, TICKS_PER_GAP, TICKS_PER_GAP,
    max_dataset_size=1000, seed=1
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=CollateCOO(DEVICE))

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
print("LR {}".format(scheduler.get_lr()))

crit = nn.BCEWithLogitsLoss()

net.train()
train_iter = iter(dataloader)

t0 = time.time()
for i in range(10000):
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
    optimizer.step()
    t2 = time.time() - t0

    if (i + 1) % 10 == 0:
        t_iter = time.time() - t0
        t0 = time.time()
        print("Iter: {}, Loss: {:.3f}, Time: {:.3f}".format(i + 1, loss.item(), t_iter))

    if (i + 1) % 250 == 0:
        print(mask_x)
        print(mask_z)
        plot_ndlar_voxels(
            [ [ el.item() for el in row ] for row in s_pred.C ],
            [ row.item() for row in s_pred.F ],
            detector, structure=False, projections=True
        )
        scheduler.step()
        print("LR {}".format(scheduler.get_lr()))

