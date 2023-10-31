from abc import ABC, abstractmethod
from collections import namedtuple

import MinkowskiEngine as ME
import torch; import torch.nn as nn; from torch.nn.utils.rnn import pad_sequence
from chamfer_distance import ChamferDistance


def init_loss_func(conf):
    if (
        conf.loss_func == "PixelWise_L1Loss" or
        conf.loss_func == "PixelWise_MSELoss" or
        conf.loss_func == "PixelWise_BCEWithLogitsLoss"
    ):
        loss = PixelWise(conf)
    elif conf.loss_func == "GapWise_L1Loss" or conf.loss_func == "GapWise_MSELoss":
        loss = GapWise(conf)
    elif conf.loss_func == "Chamfer":
        loss = Chamfer(conf)
    else:
        raise ValueError("loss_func={} not valid".format(conf.loss_func))

    return loss


class BCE:
    def __init__(self, reduction="mean"):
        self.crit = nn.BCELoss(reduction=reduction)

    def __call__(self, pred, target):
        return self.crit(pred, target)


class CustomLoss(ABC):
    @abstractmethod
    def __init__(self, conf, override_opts={}):
        pass

    @abstractmethod
    def get_names_scalefactors(self):
        pass

    @abstractmethod
    def calc_loss(self, s_pred, s_in, s_target, data):
        pass


class PixelWise(CustomLoss):
    def __init__(self, conf, override_opts={}):
        if override_opts:
            conf_dict = conf._asdict()
            for opt_name, opt_val in override_opts.items():
                conf_dict[opt_name] = opt_val
            conf_namedtuple = namedtuple("config", conf_dict)
            conf = conf_namedtuple(**conf_dict)

        if conf.loss_func == "PixelWise_L1Loss":
            self.crit = nn.L1Loss()
            self.crit_sumreduction = nn.L1Loss(reduction="sum")
        elif conf.loss_func == "PixelWise_MSELoss":
            self.crit = nn.MSELoss()
            self.crit_sumreduction = nn.MSELoss(reduction="sum")
        elif conf.loss_func == "PixelWise_BCEWithLogitsLoss":
            self.crit = nn.BCEWithLogitsLoss()
            self.crit_sumreduction = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise NotImplementedError("loss_func={} not valid".format(conf.loss_func))

        self.lambda_loss_infill_zero = conf.loss_infill_zero_weight
        self.lambda_loss_infill_nonzero = conf.loss_infill_nonzero_weight
        self.lambda_loss_active_zero = conf.loss_active_zero_weight
        self.lambda_loss_active_nonzero = conf.loss_active_nonzero_weight
        self.lambda_loss_infill = conf.loss_infill_weight
        self.lambda_loss_active = conf.loss_active_weight
        self.lambda_loss_infill_sum = conf.loss_infill_sum_weight

    def get_names_scalefactors(self):
        return {
            "infill_zero" : self.lambda_loss_infill_zero,
            "infill_nonzero" : self.lambda_loss_infill_nonzero,
            "active_zero" : self.lambda_loss_active_zero,
            "active_nonzero" : self.lambda_loss_active_nonzero,
            "infill" : self.lambda_loss_infill,
            "active" : self.lambda_loss_active,
            "infill_sum" : self.lambda_loss_infill_sum
        }

    def calc_loss(self, s_pred, s_in, s_target, data):
        (
            infill_coords, active_coords,
            infill_coords_zero, infill_coords_nonzero,
            active_coords_zero, active_coords_nonzero
        ) = self._get_infill_active_coords(s_in, s_target)

        loss_tot = 0.0
        losses = {}
        if self.lambda_loss_infill_zero:
            loss_infill_zero = self._get_loss_at_coords(s_pred, s_target, infill_coords_zero)
            loss_tot += self.lambda_loss_infill_zero * loss_infill_zero
            losses["infill_zero"] = loss_infill_zero
        if self.lambda_loss_infill_nonzero:
            loss_infill_nonzero = self._get_loss_at_coords(s_pred, s_target, infill_coords_nonzero)
            loss_tot += self.lambda_loss_infill_nonzero * loss_infill_nonzero
            losses["infill_nonzero"] = loss_infill_nonzero
        if self.lambda_loss_active_zero:
            loss_active_zero = self._get_loss_at_coords(s_pred, s_target, active_coords_zero)
            loss_tot += self.lambda_loss_active_zero * loss_active_zero
            losses["active_zero"] = loss_active_zero
        if self.lambda_loss_active_nonzero:
            loss_active_nonzero = self._get_loss_at_coords(s_pred, s_target, active_coords_nonzero)
            loss_tot += self.lambda_loss_active_nonzero * loss_active_nonzero
            losses["active_nonzero"] = loss_active_nonzero
        if self.lambda_loss_infill:
            loss_infill = self._get_loss_at_coords(s_pred, s_target, infill_coords)
            loss_tot += self.lambda_loss_infill * loss_infill
            losses["infill"] = loss_infill
        if self.lambda_loss_active:
            loss_active = self._get_loss_at_coords(s_pred, s_target, active_coords)
            loss_tot += self.lambda_loss_active * loss_active
            losses["active"] = loss_active
        if self.lambda_loss_infill_sum:
            loss_infill_sum = self._get_summed_loss_at_coords(s_pred, s_target, infill_coords)
            loss_tot += self.lambda_loss_infill_sum * loss_infill_sum
            losses["infill_sum"] = loss_infill_sum

        return loss_tot, losses

    def _get_infill_active_coords(self, s_in, s_target):
        s_in_infill_mask = s_in.F[:, -1] == 1
        infill_coords = s_in.C[s_in_infill_mask].type(torch.float)
        active_coords = s_in.C[~s_in_infill_mask].type(torch.float)

        infill_coords_zero_mask = s_target.features_at_coordinates(infill_coords)[:, 0] == 0
        infill_coords_zero = infill_coords[infill_coords_zero_mask]
        infill_coords_nonzero = infill_coords[~infill_coords_zero_mask]

        active_coords_zero_mask = s_target.features_at_coordinates(active_coords)[:, 0] == 0
        active_coords_zero = active_coords[active_coords_zero_mask]
        active_coords_nonzero = active_coords[~active_coords_zero_mask]

        return (
            infill_coords, active_coords,
            infill_coords_zero, infill_coords_nonzero,
            active_coords_zero, active_coords_nonzero
        )

    def _get_loss_at_coords(self, s_pred, s_target, coords):
        if coords.shape[0]:
            loss = self.crit(
                s_pred.features_at_coordinates(coords).squeeze(),
                s_target.features_at_coordinates(coords).squeeze()
            )
        else:
            loss = self.crit_sumreduction(
                s_pred.features_at_coordinates(coords).squeeze(),
                s_target.features_at_coordinates(coords).squeeze()
            )

        return loss

    def _get_summed_loss_at_coords(self, s_pred, s_target, coords):
        if coords.shape[0]:
            loss = self.crit(
                s_pred.features_at_coordinates(coords).squeeze().sum(),
                s_target.features_at_coordinates(coords).squeeze().sum()
            ) / len(coords)
        else:
            loss = self.crit_sumreduction(
                s_pred.features_at_coordinates(coords).squeeze().sum(),
                s_target.features_at_coordinates(coords).squeeze().sum()
            ) / len(coords)

        return loss


class GapWise(CustomLoss):
    """
    Loss on the summed adc + summed active pixels in planes of x or z = const in infill regions
    """
    def __init__(self, conf, override_opts={}):
        if override_opts:
            conf_dict = conf._asdict()
            for opt_name, opt_val in override_opts.items():
                conf_dict[opt_name] = opt_val
            conf_namedtuple = namedtuple("config", conf_dict)
            conf = conf_namedtuple(**conf_dict)

        if conf.loss_func == "GapWise_L1Loss":
            self.crit_adc = nn.L1Loss()
            self.crit_adc_sumreduction = nn.L1Loss(reduction="sum")
            self.crit_npixel = nn.L1Loss()
        elif conf.loss_func == "GapWise_MSELoss":
            self.crit_adc = nn.MSELoss()
            self.crit_adc_sumreduction = nn.MSELoss(reduction="sum")
            self.crit_npixel = nn.MSELoss()
        # elif conf.loss_func == "GapWise_L1Loss_BCELoss":
        #     self.crit_adc = nn.L1Loss()
        #     self.crit_adc_sumreduction = nn.L1Loss(reduction="sum")
        #     self.crit_npixel = nn.BCELoss()

        self.adc_threshold = conf.adc_threshold

        self.lambda_loss_infill_zero = conf.loss_infill_zero_weight
        self.lambda_loss_infill_nonzero = conf.loss_infill_nonzero_weight
        self.lambda_loss_x_gap_planes_adc = conf.loss_x_gap_planes_adc_weight
        self.lambda_loss_x_gap_planes_npixel = conf.loss_x_gap_planes_npixel_weight
        self.lambda_loss_z_gap_planes_adc = conf.loss_z_gap_planes_adc_weight
        self.lambda_loss_z_gap_planes_npixel = conf.loss_z_gap_planes_npixel_weight

    def get_names_scalefactors(self):
        return {
            "infill_zero" : self.lambda_loss_infill_zero,
            "infill_nonzero" : self.lambda_loss_infill_nonzero,
            "x_gap_planes_adc" : self.lambda_loss_x_gap_planes_adc,
            "x_gap_planes_npixel" : self.lambda_loss_x_gap_planes_npixel,
            "z_gap_planes_adc" : self.lambda_loss_z_gap_planes_adc,
            "z_gap_planes_npixel" : self.lambda_loss_z_gap_planes_npixel
        }

    def calc_loss(self, s_pred, s_in, s_target, data):
        infill_coords, infill_coords_zero, infill_coords_nonzero = self._get_infill_coords(
            s_in, s_target
        )

        loss_tot = 0.0
        losses = {}

        if self.lambda_loss_infill_zero:
            loss_infill_zero = self._get_loss_at_coords(s_pred, s_target, infill_coords_zero)
            loss_tot += self.lambda_loss_infill_zero * loss_infill_zero
            losses["infill_zero"] = loss_infill_zero
        if self.lambda_loss_infill_nonzero:
            loss_infill_nonzero = self._get_loss_at_coords(s_pred, s_target, infill_coords_nonzero)
            loss_tot += self.lambda_loss_infill_nonzero * loss_infill_nonzero
            losses["infill_nonzero"] = loss_infill_nonzero

        x_gap_losses_adc, x_gap_losses_npixel = [], []
        z_gap_losses_adc, z_gap_losses_npixel = [], []
        for i_batch in range(len(data["mask_x"])):
            batch_infill_coords = infill_coords[infill_coords[:, 0] == i_batch]

            x_gap_losses_adc_batch, x_gap_losses_npixel_batch = self._get_gap_losses(
                s_pred, s_target,
                set(data["mask_x"][i_batch]),
                batch_infill_coords,
                1,
                skip_adc=(not self.lambda_loss_x_gap_planes_adc),
                skip_npixels=(not self.lambda_loss_x_gap_planes_npixel)
            )
            x_gap_losses_adc.append(x_gap_losses_adc_batch)
            x_gap_losses_npixel.append(x_gap_losses_npixel_batch)

            z_gap_losses_adc_batch, z_gap_losses_npixel_batch = self._get_gap_losses(
                s_pred, s_target,
                set(data["mask_z"][i_batch]),
                batch_infill_coords,
                3,
                skip_adc=(not self.lambda_loss_z_gap_planes_adc),
                skip_npixels=(not self.lambda_loss_z_gap_planes_npixel)
            )
            z_gap_losses_adc.append(z_gap_losses_adc_batch)
            z_gap_losses_npixel.append(z_gap_losses_npixel_batch)

        if self.lambda_loss_x_gap_planes_adc:
            if any(x_gap_losses_adc):
                loss_x_gap_planes_adc = sum(
                    (
                        torch.mean(torch.cat(x_gap_losses_adc_batch, 0)) /
                        len(x_gap_losses_adc_batch)
                    )
                    for x_gap_losses_adc_batch in x_gap_losses_adc
                        if x_gap_losses_adc_batch
                ) / len(x_gap_losses_adc)
            else:
                loss_x_gap_planes_adc = torch.tensor(0.0)
            loss_tot += self.lambda_loss_x_gap_planes_adc * loss_x_gap_planes_adc
            losses["x_gap_planes_adc"] = loss_x_gap_planes_adc
        if self.lambda_loss_x_gap_planes_npixel:
            if any(x_gap_losses_npixel):
                loss_x_gap_planes_npixel = sum(
                    (
                        torch.mean(torch.cat(x_gap_losses_npixel_batch, 0)) /
                        len(x_gap_losses_npixel_batch)
                    )
                    for x_gap_losses_npixel_batch in x_gap_losses_npixel
                        if x_gap_losses_npixel_batch
                )
            else:
                loss_x_gap_planes_npixel = torch.tensor(0.0)
            loss_tot += self.lambda_loss_x_gap_planes_npixel * loss_x_gap_planes_npixel
            losses["x_gap_planes_npixel"] = loss_x_gap_planes_npixel
        if self.lambda_loss_z_gap_planes_adc:
            if any(z_gap_losses_adc):
                loss_z_gap_planes_adc = sum(
                    (
                        torch.mean(torch.cat(z_gap_losses_adc_batch, 0)) /
                        len(z_gap_losses_adc_batch)
                    )
                    for z_gap_losses_adc_batch in z_gap_losses_adc
                        if z_gap_losses_adc_batch
                )
            else:
                loss_z_gap_planes_adc = torch.tensor(0.0)
            loss_tot += self.lambda_loss_z_gap_planes_adc * loss_z_gap_planes_adc
            losses["z_gap_planes_adc"] = loss_z_gap_planes_adc
        if self.lambda_loss_z_gap_planes_npixel:
            if any(z_gap_losses_npixel):
                loss_z_gap_planes_npixel = sum(
                    (
                        torch.mean(torch.cat(z_gap_losses_npixel_batch, 0)) /
                        len(z_gap_losses_npixel_batch)
                    )
                    for z_gap_losses_npixel_batch in z_gap_losses_npixel
                        if z_gap_losses_npixel_batch
                )
            else:
                loss_z_gap_planes_npixel = torch.tensor(0.0)
            loss_tot += self.lambda_loss_z_gap_planes_npixel * loss_z_gap_planes_npixel
            losses["z_gap_planes_npixel"] = loss_z_gap_planes_npixel

        return loss_tot, losses

    def _get_infill_coords(self, s_in, s_target):
        s_in_infill_mask = s_in.F[:, -1] == 1
        infill_coords = s_in.C[s_in_infill_mask].type(torch.float)

        infill_coords_zero_mask = s_target.features_at_coordinates(infill_coords)[:, 0] == 0
        infill_coords_zero = infill_coords[infill_coords_zero_mask]
        infill_coords_nonzero = infill_coords[~infill_coords_zero_mask]

        return infill_coords, infill_coords_zero, infill_coords_nonzero

    def _get_loss_at_coords(self, s_pred, s_target, coords):
        if coords.shape[0]:
            loss = self.crit_adc(
                s_pred.features_at_coordinates(coords).squeeze(),
                s_target.features_at_coordinates(coords).squeeze()
            )
        else:
            loss = self.crit_adc_sumreduction(
                s_pred.features_at_coordinates(coords).squeeze(),
                s_target.features_at_coordinates(coords).squeeze()
            )

        return loss

    def _get_gap_losses(
        self, s_pred, s_target, gaps, infill_coords, coord_idx, skip_adc=False, skip_npixels=False
    ):
        gap_losses_adc, gap_losses_npixel = [], []

        if skip_adc and skip_npixels:
            return gap_losses_adc, gap_losses_npixel

        gap_ranges = self._get_edge_ranges(
            [
                int(gap_coord)
                for gap_coord in torch.unique(
                    infill_coords[:, coord_idx]
                ).tolist()
                    if int(gap_coord) in gaps
            ]
        )
        for gap_start, gap_end in gap_ranges:
            gap_mask = sum(
                infill_coords[:, coord_idx] == gap_coord
                for gap_coord in range(gap_start, gap_end + 1)
            )
            gap_coords = infill_coords[gap_mask.type(torch.bool)]

            if not skip_adc:
                gap_losses_adc.append(
                    self.crit_adc(
                        s_pred.features_at_coordinates(gap_coords).squeeze().sum(),
                        s_target.features_at_coordinates(gap_coords).squeeze().sum()
                    ).view(1) / # 0d -> 1d for cat operation later
                    len(gap_coords)
                )

            if not skip_npixels:
                gap_losses_npixel.append(
                    self.crit_npixel(
                        (
                            torch.clamp(
                                s_pred.features_at_coordinates(gap_coords).squeeze(),
                                min=0.0, max=self.adc_threshold
                            ).sum(dtype=s_pred.F.dtype) /
                            len(gap_coords) *
                            (1 / self.adc_threshold)
                        ),
                        (
                            torch.clamp(
                                s_target.features_at_coordinates(gap_coords).squeeze(),
                                min=0.0, max=self.adc_threshold
                            ).sum(dtype=s_target.F.dtype) /
                            len(gap_coords) *
                            (1 / self.adc_threshold)
                        )
                    ).view(1)
                )

        return gap_losses_adc, gap_losses_npixel

    @staticmethod
    def _get_edge_ranges(nums):
        nums = sorted(set(nums))
        discontinuities = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(discontinuities, []) + nums[-1:])
        return list(zip(edges, edges))


# NOTE I dont know how to get this working, in current state seems to compute Chamfer distance
# correctly but there are no gradients making it through. Need a way to use the features of the
# sparse tensors since these what have the gradients
class Chamfer(CustomLoss):
    """
    Loss based around Chamfer loss for pointclouds
    """
    def __init__(self, conf, override_opts={}):
        if override_opts:
            conf_dict = conf._asdict()
            for opt_name, opt_val in override_opts.items():
                conf_dict[opt_name] = opt_val
            conf_namedtuple = namedtuple("config", conf_dict)
            conf = conf_namedtuple(**conf_dict)

        self.chamfer_distance = ChamferDistance()

        self.device = torch.device(conf.device)

        self.lambda_loss_infill_chamfer = conf.loss_infill_chamfer_weight

    def get_names_scalefactors(self):
        return { "infill_chamfer" : self.lambda_loss_infill_chamfer }

    def calc_loss(self, s_pred, s_in, s_target, data):
        loss_tot = 0.0
        losses = {}

        infill_coords_nonzero_pred, infill_coords_nonzero_target = self._get_infill_coords(
            s_in, s_target, s_pred
        )

        x = [
            infill_coords_nonzero_pred[infill_coords_nonzero_pred[:,0] == i_batch][:,1:].type(torch.float)
            for i_batch in range(len(data["mask_x"]))
        ]
        x_lengths = torch.tensor([ b.shape[0] for b in x ], device=self.device, dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        y = [
            infill_coords_nonzero_target[infill_coords_nonzero_target[:,0] == i_batch][:,1:].type(torch.float)
            for i_batch in range(len(data["mask_x"]))
        ]
        y_lengths = torch.tensor([ b.shape[0] for b in y ], device=self.device, dtype=torch.long)
        y = pad_sequence(y, batch_first=True)

        # print(x)
        # print(x.shape)
        # print(y)
        # print(y.shape)
        cham_dist_1, cham_dist_2, _, _ = self.chamfer_distance(
            x, y, x_lengths=x_lengths, y_lengths=y_lengths
        )
        # print(cham_dist_1)
        # print(cham_dist_1.shape)
        # print(cham_dist_2)
        # print(cham_dist_2.shape)
        loss_infill_chamfer = torch.mean(cham_dist_1) + torch.mean(cham_dist_2)
        # print(loss_infill_chamfer)
        loss_tot += self.lambda_loss_infill_chamfer * loss_infill_chamfer
        losses["infill_chamfer"] = loss_infill_chamfer

        return loss_tot, losses

    def _get_infill_coords(self, s_in, s_target, s_pred):
        s_in_infill_mask = s_in.F[:, -1] == 1
        infill_coords = s_in.C[s_in_infill_mask]
        infill_coords_float = infill_coords.type(torch.float)

        infill_coords_nonzero_target_mask = (
            s_target.features_at_coordinates(infill_coords_float)[:, 0] != 0
        )
        infill_coords_nonzero_target = infill_coords[infill_coords_nonzero_target_mask]

        infill_coords_nonzero_pred_mask = (
            s_pred.features_at_coordinates(infill_coords_float)[:, 0] != 0
        )
        infill_coords_nonzero_pred = infill_coords[infill_coords_nonzero_pred_mask]

        return infill_coords_nonzero_pred, infill_coords_nonzero_target

