import MinkowskiEngine as ME
import torch


def make_infill_mask(s_pred, s_target, x_mask, z_mask):
    all_coords = ME.MinkowskiUnion()(s_pred, s_target).C.type(torch.int)

    infill_mask = torch.zeros(all_coords.shape[0], dtype=torch.bool)
    for i_coord, coord in enumerate(all_coords):
        b_idx = coord[0].item()
        if coord[1].item() in x_mask[b_idx] or coord[3].item() in z_mask[b_idx]:
            infill_mask[i_coord] = True

    all_coords = all_coords.type(torch.float)

    infill_coords = all_coords[infill_mask]
    infill_coords_nonzero_mask = s_target.features_at_coordinates(infill_coords)[:, 0] != 0
    infill_coords_nonzero = infill_coords[infill_coords_nonzero_mask]
    infill_coords_zero = infill_coords[~infill_coords_nonzero_mask]

    active_coords = all_coords[~infill_mask]
    active_coords_nonzero_mask = s_target.features_at_coordinates(active_coords)[:, 0] != 0
    active_coords_nonzero = active_coords[active_coords_nonzero_mask]
    active_coords_zero = active_coords[~active_coords_nonzero_mask]


def losses_gen_mask(
    s_pred, s_target, x_mask, z_mask, crit, crit_zeromask,
    loss_infill_zero=False, loss_infill_nonzero=False,
    loss_active_zero=False, loss_active_nonzero=False,
    loss_active_n_points=False, loss_infill_n_points=False,
    crit_n_points=None, crit_n_points_zeromask=None
):
    pass

