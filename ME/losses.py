from abc import ABC, abstractmethod

import MinkowskiEngine as ME
import torch; import torch.nn as nn


def init_loss_func(conf):
    if conf.loss_func == "PixelWise_L1Loss" or conf.loss_func == "PixelWise_MSELoss":
        loss = PixelWise(conf)
    elif conf.loss_func == "GapWise_L1Loss" or conf.loss_func == "GapWise_MSELoss":
        loss = GapWise(conf)
    else:
        raise ValueError("loss_func={} not valid".format(conf.loss_func))

    return loss


class CustomLoss(ABC):
    @abstractmethod
    def __init__(self, conf):
        pass

    @abstractmethod
    def get_names_scalefactors(self):
        pass

    @abstractmethod
    def calc_loss(self, s_pred, s_in, s_target, data):
        pass


class PixelWise(CustomLoss):
    def __init__(self, conf):
        if conf.loss_func == "PixelWise_L1Loss":
            self.crit = nn.L1Loss()
            self.crit_sumreduction = nn.L1Loss(reduction="sum")
        elif conf.loss_func == "PixelWise_MSELoss":
            self.crit = nn.MSELoss()
            self.crit_sumreduction = nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError("loss_func={} not valid".format(conf.loss_func))

        self.lambda_loss_infill_zero = conf.loss_infill_zero_weight
        self.lambda_loss_infill_nonzero = conf.loss_infill_nonzero_weight
        self.lambda_loss_active_zero = conf.loss_active_zero_weight
        self.lambda_loss_active_nonzero = conf.loss_active_nonzero_weight
        self.lambda_loss_infill = conf.loss_infill_weight
        self.lambda_loss_active = conf.loss_active_weight

    def get_names_scalefactors(self):
        return {
            "infill_zero" : self.lambda_loss_infill_zero,
            "infill_nonzero" : self.lambda_loss_infill_nonzero,
            "active_zero" : self.lambda_loss_active_zero,
            "active_nonzero" : self.lambda_loss_active_nonzero,
            "infill" : self.lambda_loss_infill,
            "active" : self.lambda_loss_active
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


class GapWise(CustomLoss):
    """
    Loss on the summed adc + summed active pixels in planes of x or z = const in infill regions
    """
    def __init__(self, conf):
        if conf.loss_func == "GapWise_L1Loss":
            self.crit = nn.L1Loss()
            self.crit_sumreduction = nn.L1Loss(reduction="sum")
        elif conf.loss_func == "GapWise_MSELoss":
            self.crit = nn.MSELoss()
            self.crit_sumreduction = nn.MSELoss(reduction="sum")

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

        x_gap_losses_adc, x_gap_losses_npixel = self._get_gap_losses(
            s_pred, s_target, set(data["mask_x"][0]), infill_coords, 1,
            skip_adc=(not self.lambda_loss_x_gap_planes_adc),
            skip_npixels=(not self.lambda_loss_x_gap_planes_npixel)
        )
        z_gap_losses_adc, z_gap_losses_npixel = self._get_gap_losses(
            s_pred, s_target, set(data["mask_z"][0]), infill_coords, 3,
            skip_adc=(not self.lambda_loss_z_gap_planes_adc),
            skip_npixels=(not self.lambda_loss_z_gap_planes_npixel)
        )

        # print()
        # print(x_gap_losses_adc)
        # print(x_gap_losses_npixel)
        # print()

        if self.lambda_loss_x_gap_planes_adc:
            if x_gap_losses_adc:
                loss_x_gap_planes_adc = torch.mean(torch.cat(x_gap_losses_adc, 0))
            else:
                loss_x_gap_planes_adc = torch.tensor(0.0)
            loss_tot += self.lambda_loss_x_gap_planes_adc * loss_x_gap_planes_adc
            losses["x_gap_planes_adc"] = loss_x_gap_planes_adc
        if self.lambda_loss_x_gap_planes_npixel:
            if x_gap_losses_npixel:
                loss_x_gap_planes_npixel = torch.mean(torch.cat(x_gap_losses_npixel, 0))
            else:
                loss_x_gap_planes_npixel = torch.tensor(0.0)
            loss_tot += self.lambda_loss_x_gap_planes_npixel * loss_x_gap_planes_npixel
            losses["x_gap_planes_npixel"] = loss_x_gap_planes_npixel
        if self.lambda_loss_z_gap_planes_adc:
            if z_gap_losses_adc:
                loss_z_gap_planes_adc = torch.mean(torch.cat(z_gap_losses_adc, 0))
            else:
                loss_z_gap_planes_adc = torch.tensor(0.0)
            loss_tot += self.lambda_loss_z_gap_planes_adc * loss_z_gap_planes_adc
            losses["z_gap_planes_adc"] = loss_z_gap_planes_adc
        if self.lambda_loss_z_gap_planes_npixel:
            if z_gap_losses_npixel:
                loss_z_gap_planes_npixel = torch.mean(torch.cat(z_gap_losses_npixel, 0))
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

    def _get_gap_losses(
        self, s_pred, s_target, gaps, infill_coords, coord_idx, skip_adc=False, skip_npixels=False
    ):
        gap_losses_adc, gap_losses_npixel = [], []

        if skip_adc and skip_npixels:
            return gap_losses_adc, gap_losses_npixel

        for gap in torch.unique(infill_coords[:, coord_idx]).tolist():
            if gap not in gaps:
                continue
            gap_coords = infill_coords[infill_coords[:, coord_idx] == gap]

            if not skip_adc:
                gap_losses_adc.append(
                    self.crit(
                        s_pred.features_at_coordinates(gap_coords).squeeze().sum(),
                        s_target.features_at_coordinates(gap_coords).squeeze().sum()
                    ).view(1) # 0d -> 1d for cat operation later
                )

            if not skip_npixels:
                gap_losses_npixel.append(
                    self.crit(
                        (
                            torch.clamp(
                                s_pred.features_at_coordinates(gap_coords).squeeze(),
                                min=0.0, max=self.adc_threshold
                            ).sum(dtype=s_pred.F.dtype) *
                            (1 / self.adc_threshold)
                        ),
                        (
                            torch.clamp(
                                s_target.features_at_coordinates(gap_coords).squeeze(),
                                min=0.0, max=self.adc_threshold
                            ).sum(dtype=s_target.F.dtype) *
                            (1 / self.adc_threshold)
                        )
                    ).view(1)
                )

        return gap_losses_adc, gap_losses_npixel



