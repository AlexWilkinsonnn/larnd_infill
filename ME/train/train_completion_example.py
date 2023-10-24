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
from aux import plot_ndlar_voxels_2

def main(args):
    conf = get_config(args.config)

    device = torch.device(conf.device)

    net = CompletionNet()
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

    net.train()

    crit = nn.BCEWithLogitsLoss()

    t0 = time.time()
    loss_acc = []
    n_iter = 0

    write_log_str(conf.checkpoint_dir, "Iters per epoch: {}".format(len(dataloader_train)))

    for epoch in range(conf.epochs):
        write_log_str(conf.checkpoint_dir, "==== Epoch {} ====".format(epoch))

        # Training loop
        for n_iter_epoch, data in enumerate(dataloader_train):
            optimizer.zero_grad()

            in_feat = torch.ones((len(data["input_coords"]), 1))

            s_in = ME.SparseTensor(
                features=in_feat,
                coordinates=data["input_coords"],
                device=device
            )

            cm = s_in.coordinate_manager
            target_key, _ = cm.insert_and_map(
                data["target_coords"].to(device),
                string_id="target"
            )

            # Generate from a dense tensor
            out_cls, targets, s_pred = net(s_in, target_key)
            num_layers, loss = len(out_cls), 0
            losses = []
            for out_cl, target in zip(out_cls, targets):
                curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
                losses.append(curr_loss.item())
                loss += curr_loss / num_layers

            loss.backward()
            optimizer.step()
            loss_acc.append(loss.item())

            if (
                args.print_iter and
                not isinstance(args.print_iter, str) and
                (n_iter + 1) % args.print_iter == 0
            ):
                t_iter = time.time() - t0
                t0 = time.time()
                loss_str = "epoch: {}, iter: {}, total iter: {}, ".format(
                    epoch, n_iter_epoch + 1, n_iter + 1
                )
                loss_str += "time: {:.7f}, ".format(t_iter)
                loss_str += "loss: {:.7f}".format(np.mean(loss_acc))
                write_log_str(conf.checkpoint_dir, loss_str)
                loss_acc = []
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
            loss_str = "epoch: {}, iter: {}, total iter: {}, ".format(
                epoch, n_iter_epoch + 1, n_iter + 1
            )
            loss_str += "time: {:.7f}, ".format(t_iter)
            loss_str += "loss: {:.7f}".format(np.mean(loss_acc))
            write_log_str(conf.checkpoint_dir, loss_str)
            loss_acc = []

        t0_valid = time.time()
        net.eval()
        write_log_str(conf.checkpoint_dir, "== Validation Loop ==")
        loss_acc_valid = []
        for data in tqdm(dataloader_valid, desc="Val Loop"):
            in_feat = torch.ones((len(data["input_coords"]), 1))

            s_in = ME.SparseTensor(
                feats=in_feat,
                coords=data["input_coords"],
                device=device
            )

            # Generate target sparse tensor
            cm = s_in.coords_man
            target_key = cm.create_coords_key(
                data["target_coords"],
                force_creation=True,
                allow_duplicate_coords=True,
            )

            # Generate from a dense tensor
            out_cls, targets, s_pred = net(s_in, target_key)
            num_layers, loss = len(out_cls), 0
            for out_cl, target in zip(out_cls, targets):
                loss += (
                    crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
                    / num_layers
                )
            loss_acc_valid.append(loss.item())

        t_valid = time.time() - t0_valid
        loss_str = "Validation with {} images:\n".format(len(dataset_valid))
        loss_str += "time: {:.7f}, ".format(t_valid)
        loss_str += "loss: {:.7f}".format(np.mean(loss_acc_valid))
        write_log_str(conf.checkpoint_dir, loss_str)
        loss_acc_valid = []

        # Plot last prediction of validation loop
        s_target = ME.SparseTensor(
            coordinates=data["target_coords"],
            features=torch.ones((len(data["target_coords"]), 1)),
            device=device
        )
        s_pred = ME.SparseTensor(
            features=torch.ones((len(s_pred.C), 1)),
            coordinates=s_pred.C,
            device=device
        )
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


class CompletionNet(nn.Module):

    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, in_nchannel=512):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, partial_in, target_key):
        out_cls, targets = [], []

        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        target = self.get_target(dec_s32, target_key)
        targets.append(target)
        out_cls.append(dec_s32_cls)

        if self.training:
            keep_s32 += target

        # Remove voxels s32
        dec_s32 = self.pruning(dec_s32, keep_s32)

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        target = self.get_target(dec_s16, target_key)
        targets.append(target)
        out_cls.append(dec_s16_cls)

        if self.training:
            keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)

        target = self.get_target(dec_s8, target_key)
        targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        if self.training:
            keep_s8 += target

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        target = self.get_target(dec_s4, target_key)
        targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        if self.training:
            keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        target = self.get_target(dec_s2, target_key)
        targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        if self.training:
            keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        target = self.get_target(dec_s1, target_key)
        targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)

        return out_cls, targets, dec_s1


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument(
        "--print_iter", type=int, default=200, help="zero for never, -1 for every epoch"
    )

    args = parser.parse_args()

    args.print_iter = "epoch" if args.print_iter == -1 else args.print_iter

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

