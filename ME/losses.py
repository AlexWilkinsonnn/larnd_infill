import MinkowskiEngine as ME
import torch; import torch.nn as nn


def init_loss_func(conf):
    if conf.loss_func == "PixelWise_L1Loss":
        loss = PixelWise(
            "L1Loss",
            conf.loss_infill_zero_weight, conf.loss_infill_nonzero_weight,
            conf.loss_active_zero_weight, conf.loss_active_nonzero_weight
        )
    elif conf.loss_func == "PixelWise_MSELoss":
        loss = PixelWise(
            "MSELoss",
            conf.loss_infill_zero_weight, conf.loss_infill_nonzero_weight,
            conf.loss_active_zero_weight, conf.loss_active_nonzero_weight
        )
    else:
        raise NotImplementedError("loss_func={} not valid".format(conf.loss_func))

    return loss


class PixelWise:
    def __init__(
        self,
        loss_func,
        lambda_loss_infill_zero, lambda_loss_infill_nonzero,
        lambda_loss_active_zero, lambda_loss_active_nonzero
    ):
        if loss_func == "L1Loss":
            self.crit = nn.L1Loss()
            self.crit_sumreduction = nn.L1Loss(reduction="sum")
        elif loss_func == "MSELoss":
            self.crit = nn.MSELoss()
            self.crit_sumreduction = nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError("loss_func={} not valid".format(loss_func))

        self.lambda_loss_infill_zero = lambda_loss_infill_zero
        self.lambda_loss_infill_nonzero = lambda_loss_infill_nonzero
        self.lambda_loss_active_zero = lambda_loss_active_zero
        self.lambda_loss_active_nonzero = lambda_loss_active_nonzero

    def calc_loss(self, s_pred, s_in, s_target):
        ret = self._get_infill_active_coords(s_in, s_target)
        infill_coords_zero, infill_coords_nonzero, active_coords_zero, active_coords_nonzero = ret

        loss_infill_zero = self._get_loss_at_coords(s_pred, s_target, infill_coords_zero)
        loss_infill_nonzero = self._get_loss_at_coords(s_pred, s_target, infill_coords_nonzero)
        loss_active_zero = self._get_loss_at_coords(s_pred, s_target, active_coords_zero)
        loss_active_nonzero = self._get_loss_at_coords(s_pred, s_target, active_coords_nonzero)

        loss_tot = (
            self.lambda_loss_infill_zero * loss_infill_zero +
            self.lambda_loss_infill_nonzero * loss_infill_nonzero +
            self.lambda_loss_active_zero * loss_active_zero +
            self.lambda_loss_active_nonzero * loss_active_nonzero
        )

        return (
            loss_tot,
            {
                "infill_zero" : loss_infill_zero, "infill_nonzero" : loss_infill_nonzero,
                "active_zero" : loss_active_zero, "active_nonzero" : loss_active_nonzero
            }
        )

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

        return infill_coords_zero, infill_coords_nonzero, active_coords_zero, active_coords_nonzero

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


"""
Loss on number of active pixels and/or summed adc in a x=cnst slices of infill region
"""
class PlaneWise:
    def __init__(self):
        pass

    

