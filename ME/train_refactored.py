import argpars, time

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from ME.config_parser import get_config
from ME.dataset import LarndDataset, DataPrepType, CollateCOO
from ME.models.completion_net import CompletionNet, CompletionNetSigMask, MyCompletionNet


def main(args):
    config = get_config(args.config)
    device = torch.device(config.device)

    dataset = LarndDataset(
        config.data_path,
        config.data_prep_type,
        config.vmap,
        config.n_feats_in, config.n_feats_out,
        config.scalefactors,
        max_dataset_size=config.max_dataset_size,
        seed=1
    )
    
    if config.data_prep_type == DataPrepType.STANDARD:
        raise NotImplementedError()

    elif config.data_prep_type == DataPrepType.REFLECTION:
        pass

    elif config.data_prep_type == DataPrepType.GAP_DISTANCE:
        collate_fn = CollateCOO(
            coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=max(config.max_num_workers, config.batch_size),
            collate_fn=collate_fn
        )

        net = MyCompletionNet(
            (
                config.vmap["n_voxels"]["x"],
                config.vmap["n_voxels"]["y"],
                config.vmap["n_voxels"]["z"]
            ),
            in_nchannel=config.n_feats_in + 2, out_nchannel=config.n_feats_out,
            final_pruning_threshold=(1 / 150)
        )
        net.to(device)
        print(
            "Model has {:.1f} million parameters".format(
                sum(params.numel() for params in net.parameters()) / 1e6
            )
        )

        optimizer = optim.SGD(
            net.parameters(), lr=config.initial_lr, momentum=0.9, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

        crit, crit_zeromask = init_loss_func(config.loss_func)
        crit_n_points, crit_n_points_zeromask = init_loss_func(config.loss_func_n_points)



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


def train_nomask(net, epochs, optimizer, device):
    net.train()

    for i_epoch in range(epochs):
        t_0_epoch = time.time()
        for batch_data in dataloader:
            optimizer.zero_grad()

            s_in = ME.SparseTensor(
                coordinates=batch_data["input_coords"], features=batch_data["input_feats"],
                device=device
            )
            s_target = ME.SparseTensor(
                coordinates=batch_data["target_coords"], features=batch_data["target_feats"],
                device=device,
                coordinate_manager=s_in.coordinate_manager
            )

            try:
                s_pred = net(s_in)
            except ValueError as e:
                print(s_in.C)
                print(s_in.C.shape)
                print(s_in.F.shape)
                raise e

                

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument("--valid_iter", type=int, default=1000)
    parser.add_argument("--print_iter", type=int, default=200)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
