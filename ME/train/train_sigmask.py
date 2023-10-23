import time, argparse, os
from collections import defaultdict

import yaml
import numpy as np
from tqdm import tqdm

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties

from ME.config_parser import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net import CompletionNetSigMask
from ME.losses import init_loss_func
from aux import plot_ndlar_voxels_2

def main(args):
    conf = get_config(args.config)

    device = torch.device(conf.device)

    net = CompletionNetSigMask(
        (conf.vmap["n_voxels"]["x"], conf.vmap["n_voxels"]["y"], conf.vmap["n_voxels"]["z"]),
        in_nchannel=conf.n_feats_in + 1, out_nchannel=conf.n_feats_out,
        final_pruning_threshold=conf.adc_threshold
    )
    net.to(device)
    print(
        "Model has {:.1f} million parameters".format(
            sum(params.numel() for params in net.parameters()) / 1e6
        )
    )

    collate_fn = CollateCOO(
        coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    )
    dataset_train = LarndDataset(
        conf.train_data_path,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        conf.xyz_smear_infill, conf.xyz_smear_active,
        max_dataset_size=conf.max_dataset_size
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=max(conf.max_num_workers, conf.batch_size),
        shuffle=True
    )
    dataset_valid = LarndDataset(
        conf.valid_data_path,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        conf.xyz_smear_infill, conf.xyz_smear_active,
        max_dataset_size=conf.max_valid_dataset_size
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=max(conf.max_num_workers, conf.batch_size),
        shuffle=True # Shuffling to see different events in saved validation image
    )

    optimizer = optim.SGD(net.parameters(), lr=conf.initial_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    print("LR {}".format(scheduler.get_lr()))

    loss_cls = init_loss_func(conf)
    loss_scalefactors = loss_cls.get_names_scalefactors()

    net.train()

    t0 = time.time()
    losses_acc = defaultdict(list)
    n_iter = 0

    write_log_str(conf.checkpoint_dir, "Iters per epoch: {}".format(len(dataloader_train)))

    for epoch in range(conf.epochs):
        write_log_str(conf.checkpoint_dir, "==== Epoch {} ====".format(epoch))

        # Training loop
        for n_iter_epoch, data in enumerate(dataloader_train):
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

            loss_tot, losses = loss_cls.calc_loss(s_pred, s_in, s_target, data)

            loss_tot.backward()
            losses_acc["tot"].append(loss_tot.item())
            for loss_name, loss in losses.items():
                losses_acc[loss_name].append(loss.item())

            optimizer.step()

            if (
                args.print_iter and
                not isinstance(args.print_iter, str) and
                (n_iter + 1) % args.print_iter == 0
            ):
                t_iter = time.time() - t0
                t0 = time.time()
                loss_str = get_print_str(
                    epoch, losses_acc, loss_scalefactors, n_iter_epoch, n_iter, t_iter, s_pred
                )
                write_log_str(conf.checkpoint_dir, loss_str)
                losses_acc = defaultdict(list)
            if (
                args.plot_iter and
                not isinstance(args.plot_iter, str) and
                (n_iter + 1) % args.plot_iter == 0
            ):
                plot_pred(
                    s_pred, s_in, s_target,
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    "epoch{}-iter{}".format(epoch, n_iter_epoch + 1),
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoint_dir, "preds"),
                    save_tensors=True
                )
            if (
                conf.lr_decay_iter and
                not isinstance(conf.lr_decay_iter, str) and
                (n_iter + 1) % conf.lr_decay_iter == 0
            ):
                scheduler.step()
                write_log_str(conf.checkpoint_dir, "LR {}".format(scheduler.get_lr()))

            n_iter += 1

        if isinstance(conf.lr_decay_iter, str) and conf.lr_decay_iter == "epoch":
            scheduler.step()
            write_log_str(conf.checkpoint_dir, "LR {}".format(scheduler.get_lr()))
        if isinstance(args.print_iter, str) and args.print_iter == "epoch":
            t_iter = time.time() - t0
            t0 = time.time()
            loss_str = get_print_str(
                epoch, losses_acc, loss_scalefactors, n_iter_epoch, n_iter, t_iter, s_pred
            )
            write_log_str(conf.checkpoint_dir, loss_str)
            losses_acc = defaultdict(list)
        if isinstance(args.plot_iter, str) and args.plot_iter == "epoch":
            plot_pred(
                s_pred, s_in, s_target,
                data,
                conf.vmap,
                conf.scalefactors,
                "epoch{}-end".format(epoch),
                conf.detector,
                save_dir=os.path.join(conf.checkpoint_dir, "preds"),
                save_tensors=True
            )

        # Save latest network
        print("Saving net...")
        torch.save(net.cpu().state_dict(), os.path.join(conf.checkpoint_dir, "latest_net.pth"))
        net.to(device)

        # Validation loop
        write_log_str(conf.checkpoint_dir, "== Validation Loop ==")
        losses_acc_valid = defaultdict(list)
        for data in tqdm(dataloader_valid, desc="Val Loop"):
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

            with torch.no_grad():
                s_pred = net(s_in)

            loss_tot, losses = loss_cls.calc_loss(s_pred, s_in, s_target, data)

            losses_acc_valid["tot"].append(loss_tot.item())
            for loss_name, loss in losses.items():
                losses_acc[loss_name].append(loss.item())

        loss_str = (
            "Validation with {} images:\n".format(len(dataset_valid)) +
			get_loss_str(losses_acc_valid, loss_scalefactors)
        )
        write_log_str(conf.checkpoint_dir, loss_str)
        # Plot last prediction of validation loop
        plot_pred(
            s_pred, s_in, s_target,
            data,
            conf.vmap,
            conf.scalefactors,
            "epoch{}-valid".format(epoch),
            conf.detector,
            save_dir=os.path.join(conf.checkpoint_dir, "preds"),
            save_tensors=True
        )


def write_log_str(checkpoint_dir, log_str, print_str=True):
    if print_str:
        print(log_str)
    with open(os.path.join(checkpoint_dir, "losses.txt"), 'a') as f:
        f.write(log_str + '\n')


def get_print_str(epoch, losses_acc, loss_scalefactors, n_iter, n_iter_tot, t_iter, s_pred):
    return (
        "Epoch: {}, Iter: {}, Total Iter: {}, ".format(epoch, n_iter + 1, n_iter_tot + 1) +
        "Time: {:.7f}, last s_pred.shape: {}\n\t".format(t_iter, s_pred.shape) +
		get_loss_str(losses_acc, loss_scalefactors)
    )


def get_loss_str(losses_acc, loss_scalefactors):
    loss_str = "Losses: total={:.7f}".format(np.mean(losses_acc["tot"]))
    for loss_name, loss in sorted(losses_acc.items()):
        if loss_name == "tot":
            continue
        loss_u = np.mean(loss)
        loss_str += (
            " " + loss_name + "={:.7f} ({:.7f})".format(
                loss_u, loss_u * loss_scalefactors[loss_name]
            )
        )
    return loss_str


def plot_pred(
    s_pred, s_in, s_target, data, vmap, scalefactors, save_name_prefix, detector,
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
            saveas=os.path.join(save_dir, "{}_batch{}_pred.pdf".format(save_name_prefix, i_batch)),
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
                save_dir, "{}_batch{}_pred_predonly.pdf".format(save_name_prefix, i_batch)
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
            saveas=os.path.join(
                save_dir, "{}_batch{}_target.pdf".format(save_name_prefix, i_batch)
            ),
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
                os.path.join(save_dir,"{}_batch{}_in.yml".format(save_name_prefix, i_batch)), "w"
            ) as f:
                yaml.dump(in_dict, f)

            pred_dict = {
                tuple(coord.tolist()) : feat.tolist()
                for coord, feat in zip(coords_pred, feats_pred)
            }
            with open(
                os.path.join(save_dir,"{}_batch{}_pred.yml".format(save_name_prefix, i_batch)), "w"
            ) as f:
                yaml.dump(pred_dict, f)

            target_dict = {
                tuple(coord.tolist()) : feat.tolist()
                for coord, feat in zip(coords_target, feats_target)
            }
            with open(
                os.path.join(save_dir,"{}_batch{}_target.yml".format(save_name_prefix, i_batch)),
                "w"
            ) as f:
                yaml.dump(target_dict, f)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument(
        "--plot_iter", type=int, default=1000, help="zero for never, -1 of every epoch"
    )
    parser.add_argument(
        "--print_iter", type=int, default=200, help="zero for never, -1 for every epoch"
    )

    args = parser.parse_args()

    args.plot_iter = "epoch" if args.plot_iter == -1 else args.plot_iter
    args.print_iter = "epoch" if args.print_iter == -1 else args.print_iter

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

