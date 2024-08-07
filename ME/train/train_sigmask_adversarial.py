import time, argparse, os, glob

from collections import defaultdict

import yaml
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt; from matplotlib.backends.backend_pdf import PdfPages

import torch; import torch.optim as optim; import torch.nn as nn
import MinkowskiEngine as ME

from ME.config_parsers.parser_train import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net_adversarial import CompletionNetAdversarial
from ME.losses import GapWise
from aux import plot_ndlar_voxels_2

PRINT_N_PARAMS=True
PRINT_FORWARD_PASS=True

def main(args):
    conf = get_config(args.config)

    if conf.train_script != "train_sigmask_adversarial":
        raise ValueError("Cannot run this config with train_sigmask_adversarial.py!")

    model = CompletionNetAdversarial(conf)
    if PRINT_N_PARAMS:
        print(
            "Generator has {:.1f} million parameters".format(
                sum(params.numel() for params in model.net_G.parameters()) / 1e6
            )
        )
        print(
            "Discriminator has {:.1f} million parameters".format(
                sum(params.numel() for params in model.net_D.parameters()) / 1e6
            )
        )
    if PRINT_FORWARD_PASS:
        dummy_batch_coords = torch.cat([torch.zeros(500), torch.ones(500)]).unsqueeze(1)
        dummy_coords = torch.cat([dummy_batch_coords, torch.randint(0, 1000, (1000, 3))], axis=1)
        # dummy_coords = torch.cat([dummy_batch_coords] + [torch.arange(0, 1000).unsqueeze(1) for _ in range(3)], axis=1)
        dummy_in = ME.SparseTensor(
            coordinates=dummy_coords,
            features=torch.rand(1000, conf.n_feats_in + 1),
            device=model.device
        )
        model.net_G.print_forward_pass(dummy_in)
        if conf.net_D == "MEClassifier":
            dummy_in = ME.TensorField(
                coordinates=dummy_coords,
                features=torch.rand(1000, conf.n_feats_out),
                device=model.device
            )
        else:
            dummy_in = ME.SparseTensor(
                coordinates=dummy_coords,
                features=torch.rand(1000, conf.n_feats_out),
                device=model.device
            )
        model.net_D.print_forward_pass(dummy_in)
        del dummy_in

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
        conf.xyz_max_reflect_distance,
        max_dataset_size=conf.max_dataset_size
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        shuffle=True
    )
    dataset_valid = LarndDataset(
        conf.valid_data_path,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        conf.xyz_smear_infill, conf.xyz_smear_active,
        conf.xyz_max_reflect_distance,
        max_dataset_size=conf.max_valid_dataset_size,
        seed=1
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=True # Shuffling to see different events in saved validation image
    )

    print("LR G {}".format(model.scheduler_G.get_lr()))
    print("LR D {}".format(model.scheduler_D.get_lr()))

    loss_scalefactors = model.get_loss_names_weights()
    # Using methods in this for accuracy metrics in validation
    gapwise_cls = GapWise(conf, override_opts={ "loss_func" : "GapWise_L1Loss" })

    t0 = time.time()
    losses_acc = defaultdict(list)
    n_iter = 0
    prev_val_loss = float("inf")

    write_log_str(conf.checkpoint_dir, "Iters per epoch: {}".format(len(dataloader_train)))

    for epoch in range(conf.epochs):
        write_log_str(conf.checkpoint_dir, "==== Epoch {} ====".format(epoch))

        # Training loop
        if conf.refresh_masks_epoch > 1:
            if (epoch % conf.refresh_masks_epoch) == 0:
                dataloader_train.dataset.set_fill_cache()
            else:
                dataloader_train.dataset.set_use_cache()
        else:
            dataloader_train.dataset.set_cache_off()
        model.train()
        model.new_epoch(epoch)
        for n_iter_epoch, data in enumerate(dataloader_train):
            if not model.set_input(data):
                n_iter += 1
                continue

            model.optimize_parameters()

            losses = model.get_current_losses()
            for loss_name, loss in losses.items():
                losses_acc[loss_name].append(loss)

            if (
                args.print_iter and
                not isinstance(args.print_iter, str) and
                (n_iter + 1) % args.print_iter == 0
            ):
                t_iter = time.time() - t0
                t0 = time.time()
                loss_str = get_print_str(
                    epoch, losses_acc, loss_scalefactors, n_iter_epoch, n_iter, t_iter
                )
                write_log_str(conf.checkpoint_dir, loss_str)
                losses_acc = defaultdict(list)
            if (
                args.plot_iter and
                not isinstance(args.plot_iter, str) and
                (n_iter + 1) % args.plot_iter == 0
            ):
                vis = model.get_current_visuals()
                plot_pred(
                    vis["s_pred"], vis["s_in"], vis["s_target"],
                    data,
                    conf.vmap,
                    conf.scalefactors,
                    "epoch{}-iter{}".format(epoch, n_iter_epoch + 1),
                    conf.detector,
                    save_dir=os.path.join(conf.checkpoint_dir, "preds"),
                    save_tensors=True,
                    skip_predonly=True
                )
            if (
                conf.lr_decay_iter and
                not isinstance(conf.lr_decay_iter, str) and
                (n_iter + 1) % conf.lr_decay_iter == 0
            ):
                model.scheduler_step()
                write_log_str(conf.checkpoint_dir, "LR G {}".format(model.scheduler_G.get_lr()))
                write_log_str(conf.checkpoint_dir, "LR D {}".format(model.scheduler_D.get_lr()))

            n_iter += 1

        if isinstance(conf.lr_decay_iter, str) and conf.lr_decay_iter == "epoch":
            model.scheduler_step()
            write_log_str(conf.checkpoint_dir, "LR G {}".format(model.scheduler_G.get_lr()))
            write_log_str(conf.checkpoint_dir, "LR D {}".format(model.scheduler_D.get_lr()))
        if isinstance(args.print_iter, str) and args.print_iter == "epoch":
            t_iter = time.time() - t0
            t0 = time.time()
            loss_str = get_print_str(
                epoch, losses_acc, loss_scalefactors, n_iter_epoch, n_iter, t_iter
            )
            write_log_str(conf.checkpoint_dir, loss_str)
            losses_acc = defaultdict(list)
        if isinstance(args.plot_iter, str) and args.plot_iter == "epoch":
            vis = model.get_current_visuals()
            plot_pred(
                vis["s_pred"], vis["s_in"], vis["s_target"],
                data,
                conf.vmap,
                conf.scalefactors,
                "epoch{}-end".format(epoch),
                conf.detector,
                save_dir=os.path.join(conf.checkpoint_dir, "preds"),
                save_tensors=True,
                skip_predonly=True
            )

        # Save latest network
        if conf.save_model in ["latest", "all"]:
            print("Saving latest nets...")
            model.save_networks("latest")

        # Validation loop
        # Always want the same validation data
        if epoch == 0:
            dataloader_valid.dataset.set_fill_cache()
        else:
            dataloader_valid.dataset.set_use_cache()
        model.eval()
        write_log_str(conf.checkpoint_dir, "== Validation Loop ==")
        losses_acc_valid = defaultdict(list)
        losses_acc_valid_unscaled = defaultdict(list)
        x_gap_abs_diffs, x_gap_frac_diffs = [], []
        z_gap_abs_diffs, z_gap_frac_diffs = [], []
        for data in tqdm(dataloader_valid, desc="Val Loop"):
            model.set_input(data)
            model.test(compute_losses=True)

            losses = model.get_current_losses(valid=True)
            for loss_name, loss in losses.items():
                losses_acc_valid[loss_name].append(loss)

            ret = model.get_current_visuals()
            s_pred, s_target, s_in = ret["s_pred"], ret["s_target"], ret["s_in"]

            if conf.n_feats_out != 1:
                raise NotImplementedError("validation loop expected predicting only adc")
            # Get loss metrics without scalefactors applied
            s_pred_unscaled = ME.SparseTensor(
                coordinates=s_pred.C, features=s_pred.F * (1 / conf.scalefactors[0])
            )
            s_in_unscaled = ME.SparseTensor(
                coordinates=s_in.C,
                features=torch.cat(
                    (
                        [
                            s_in .F[:, [i]] * (1 * conf.scalefactors[i])
                            for i in range(conf.n_feats_in)
                        ] +
                        [ s_in.F[:, [-1]] ]
                    ),
                    dim=1
                )
            )
            s_target_unscaled = ME.SparseTensor(
                coordinates=s_target.C, features=s_target.F * (1 / conf.scalefactors[0])
            )
            loss_tot_unscaled, losses_unscaled = model.lossfunc_G.calc_loss(
                s_pred_unscaled, s_in_unscaled, s_target_unscaled, data
            )
            losses_acc_valid_unscaled["pixel_tot"].append(loss_tot_unscaled.item())
            for loss_name, loss in losses_unscaled.items():
                losses_acc_valid_unscaled[loss_name].append(loss.item())

            # Get infill accuracy metrics
            infill_coords, _, _ = gapwise_cls._get_infill_coords(s_in_unscaled, s_target_unscaled)

            for i_batch in range(len(data["mask_x"])):
                batch_infill_coords = infill_coords[infill_coords[:, 0] == i_batch]

                x_gap_ranges = gapwise_cls._get_edge_ranges(
                    [
                        int(gap_coord)
                        for gap_coord in torch.unique(
                            batch_infill_coords[:, 1]
                        ).tolist()
                            if int(gap_coord) in set(data["mask_x"][i_batch])
                    ]
                )

                for gap_start, gap_end in x_gap_ranges:
                    gap_mask = sum(
                        batch_infill_coords[:, 1] == gap_coord
                        for gap_coord in range(gap_start, gap_end + 1)
                    )
                    gap_coords = batch_infill_coords[gap_mask.type(torch.bool)]
                    pred_sum = (
                        s_pred_unscaled.features_at_coordinates(gap_coords).squeeze().sum().item()
                    )
                    target_sum = (
                        s_target_unscaled.features_at_coordinates(gap_coords).squeeze().sum().item()
                    )
                    x_gap_abs_diffs.append(pred_sum - target_sum)
                    x_gap_frac_diffs.append((pred_sum - target_sum) / max(target_sum, 1.0))

                z_gap_ranges = gapwise_cls._get_edge_ranges(
                    [
                        int(gap_coord)
                        for gap_coord in torch.unique(
                            batch_infill_coords[:, 3]
                        ).tolist()
                            if int(gap_coord) in set(data["mask_z"][i_batch])
                    ]
                )

                for gap_start, gap_end in z_gap_ranges:
                    gap_mask = sum(
                        batch_infill_coords[:, 3] == gap_coord
                        for gap_coord in range(gap_start, gap_end + 1)
                    )
                    gap_coords = batch_infill_coords[gap_mask.type(torch.bool)]
                    pred_sum = (
                        s_pred_unscaled.features_at_coordinates(gap_coords).squeeze().sum().item()
                    )
                    target_sum = (
                        s_target_unscaled.features_at_coordinates(gap_coords).squeeze().sum().item()
                    )
                    z_gap_abs_diffs.append(pred_sum - target_sum)
                    z_gap_frac_diffs.append((pred_sum - target_sum) / max(target_sum, 1.0))

        losses_acc_valid_unscaled["tot"] = losses_acc_valid["tot"]
        losses_acc_valid_unscaled["G_GAN"] = losses_acc_valid["G_GAN"]

        loss_str = (
            "Validation with {} images:\n".format(len(dataset_valid)) +
            get_loss_str(losses_acc_valid, loss_scalefactors) +
            "\nUnscaled:\n" +
            get_loss_str(losses_acc_valid_unscaled, loss_scalefactors)
        )
        write_log_str(conf.checkpoint_dir, loss_str)

        # Plot last prediction of validation loop
        vis = model.get_current_visuals()
        plot_pred(
            vis["s_pred"], vis["s_in"], vis["s_target"],
            data,
            conf.vmap,
            conf.scalefactors,
            "epoch{}-valid".format(epoch),
            conf.detector,
            save_dir=os.path.join(conf.checkpoint_dir, "preds"),
            save_tensors=True
        )

        # plot prediction using the true gap positions
        dataloader_valid.dataset.set_cache_off()
        dataloader_valid.dataset.set_use_true_gaps(True)
        data = next(iter(dataloader_valid))
        model.set_input(data)
        model.test(compute_losses=False)
        vis = model.get_current_visuals()
        plot_pred(
            vis["s_pred"], vis["s_in"], vis["s_target"],
            data,
            conf.vmap,
            conf.scalefactors,
            "epoch{}-valid-truegaps".format(epoch),
            conf.detector,
            save_dir=os.path.join(conf.checkpoint_dir, "preds"),
            save_tensors=True,
            max_evs=1,
            skip_target=True,
            skip_predonly=True
        )
        dataloader_valid.dataset.set_use_true_gaps(False)

        # save summed adc different histograms
        pdf_val_plots = PdfPages(
            os.path.join(
                conf.checkpoint_dir, "preds", "epoch{}-valid_summary_plots.pdf".format(epoch)
            )
        )
        gen_val_histo(
            x_gap_abs_diffs, 200, (-500,500), pdf_val_plots,
            "Absolute x Gap Summed ADC Differences", r"$\sum_{x gap} Pred - \sum_{x gap} True$"
        )
        gen_val_histo(
            x_gap_frac_diffs, 200, (-2.0,2.0), pdf_val_plots,
            "Absolute x Gap Summed ADC Fractional Differences",
            r"$\frac{\sum_{x gap} Pred - \sum_{xjgap} True}{\sum_{x gap} True}$"
        )
        gen_val_histo(
            z_gap_abs_diffs, 200, (-500,500), pdf_val_plots,
            "Absolute z Gap Summed ADC Differences", r"$\sum_{z gap} Pred - \sum_{z gap} True$"
        )
        gen_val_histo(
            z_gap_frac_diffs, 200, (-2.0,2.0), pdf_val_plots,
            "Absolute z Gap Summed ADC Fractional Differences",
            r"$\frac{\sum_{z gap} Pred - \sum_{z gap} True}{\sum_{z gap} True}$"
        )
        pdf_val_plots.close()

        if conf.save_model in ["best", "all"]:
            curr_val_loss = np.mean(losses_acc_valid["tot"])
            if curr_val_loss < prev_val_loss:
                print(
                    "New best loss ({} < {}), saving nets...".format(curr_val_loss, prev_val_loss)
                )
                prev_paths = glob.glob(os.path.join(conf.checkpoint_dir, "**best_epoch**.pth"))
                if len(prev_paths) > 2:
                    raise Exception("About to delete {}, really?".format(prev_paths))
                for path in prev_paths:
                    os.remove(path)
                model.save_networks("best_epoch{}".format(epoch))
                prev_val_loss = curr_val_loss

def write_log_str(checkpoint_dir, log_str, print_str=True):
    if print_str:
        print(log_str)
    with open(os.path.join(checkpoint_dir, "losses.txt"), 'a') as f:
        f.write(log_str + '\n')

def get_print_str(epoch, losses_acc, loss_scalefactors, n_iter, n_iter_tot, t_iter):
    return (
        "Epoch: {}, Iter: {}, Total Iter: {}, ".format(epoch, n_iter + 1, n_iter_tot + 1) +
        "Time: {:.7f}\n\t".format(t_iter) +
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
    max_evs=2, save_dir="test/", save_tensors=False, skip_target=False, skip_predonly=False,
    adc_threshold=4, autocrop=False
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
            if feat < adc_threshold:
                continue
            if coord[0].item() in batch_mask_x or coord[2].item() in batch_mask_z:
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats_list.append(int(feat.item()))
            coords_packed_predonly[0].append(coord[0].item())
            coords_packed_predonly[1].append(coord[1].item())
            coords_packed_predonly[2].append(coord[2].item())
            feats_list_predonly.append(int(feat.item()))
        for coord, feat in zip(coords_target, feats_target):
            if feat < adc_threshold:
                continue
            coords_target_packed[0].append(coord[0].item())
            coords_target_packed[1].append(coord[1].item())
            coords_target_packed[2].append(coord[2].item())
            feats_list_target.append(int(feat.item()))
            if coord[0].item() not in batch_mask_x and coord[2].item() not in batch_mask_z:
                coords_packed[0].append(coord[0].item())
                coords_packed[1].append(coord[1].item())
                coords_packed[2].append(coord[2].item())
                feats_list.append(int(feat.item()))

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
            signal_mask_active_coords=coords_sigmask_active_packed,
            autocrop=autocrop
        )
        if not skip_predonly:
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
                signal_mask_active_coords=coords_sigmask_active_packed,
                autocrop=autocrop
            )
        if not skip_target:
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
                signal_mask_active_coords=coords_sigmask_active_packed,
                autocrop=autocrop
            )

        if save_tensors:
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

            if not skip_target:
                target_dict = {
                    tuple(coord.tolist()) : feat.tolist()
                    for coord, feat in zip(coords_target, feats_target)
                }
                with open(
                    os.path.join(
                        save_dir,"{}_batch{}_target.yml".format(save_name_prefix, i_batch)
                    ),
                    "w"
                ) as f:
                    yaml.dump(target_dict, f)

def gen_val_histo(x, bins, range, pdf, title, xlabel):
    _, ax = plt.subplots(figsize=(12,8))
    ax.hist(x, bins=bins, range=range, histtype="step")
    ax.set_title(title, fontsize=16)
    ax.grid(visible=True)
    ax.set_xlabel(xlabel, fontsize=12)
    pdf.savefig(bbox_inches="tight")
    plt.close()

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

