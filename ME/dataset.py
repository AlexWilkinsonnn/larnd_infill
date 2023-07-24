import os, math, time
from collections import defaultdict
from enum import Enum

import numpy as np
import sparse, yaml
from tqdm import tqdm

import torch
import MinkowskiEngine as ME

from aux import plot_ndlar_voxels_2


class LarndDataset(torch.utils.data.Dataset):
    """Dataset for reading sparse volexised larnd-sim data and preparing an infill mask"""
    def __init__(
        self, dataroot, mask_type, vmap, n_feats_in, n_feats_out,
        x_true_gap_padding=None, z_true_gap_padding=None,
        valid=False, max_dataset_size=0, seed=None

    ):
        if seed is not None:
            np.random.seed(seed)

        self.mask_type = mask_type

        # Assuming the first n features are the target features
        self.n_feats_out = n_feats_out

        self.vmap = vmap
        self.x_true_gaps = np.array(vmap["x_gaps"])
        self.z_true_gaps = np.array(vmap["z_gaps"])

        self.x_gap_size = self._calc_gap_size(self.x_true_gaps)
        self.x_gap_spacing = self._calc_gap_spacing(self.x_true_gaps, vmap["n_voxels"]["x"])
        self.z_gap_size = self._calc_gap_size(self.z_true_gaps)
        self.z_gap_spacing = self._calc_gap_spacing(self.z_true_gaps, vmap["n_voxels"]["z"])

        self.x_true_gap_padding = (
            self.x_gap_size if x_true_gap_padding is None else x_true_gap_padding
        )
        self.z_true_gap_padding = (
            self.z_gap_size if z_true_gap_padding is None else z_true_gap_padding
        )

        # data_dir = os.path.join(dataroot, "valid" if valid else "train")
        data_dir = dataroot
        self.data = []
        for i, f in tqdm(
            enumerate(os.listdir(data_dir)),
            desc="Reading dataset into memory",
            total=max_dataset_size if max_dataset_size else None
        ):
            if max_dataset_size and i >= max_dataset_size:
                break

            self.data.append(dict())

            coo = sparse.load_npz(os.path.join(data_dir, f))

            # [x][y][z] = [feat_1, feat_2]
            self.data[-1]["xyz"] = self._coo2nested(coo, 0, 1, 2, n_feats_in)

            if mask_type == MaskType.REFLECTION:
                self.data[-1]["zxy"] = self._coo2nested(coo, 2, 0, 1, n_feats_in)

    """ __init__ helpers """

    @staticmethod
    def _calc_gap_size( gaps):
        gap_chunks = np.append(np.insert((np.diff(gaps) != 1).nonzero()[0], 0, 0), gaps.size - 1)

        gap_sizes = np.diff(gap_chunks)
        gap_sizes[0] += 1

        assertion = np.unique(gap_sizes).size == 1
        assert assertion, "Expected equal x gap sizes: {}, {}".format(gap_sizes, gaps)

        return gap_sizes[0]

    @staticmethod
    def _calc_gap_spacing(gaps, n_voxels):
        gap_chunks = np.append(np.insert((np.diff(gaps) != 1).nonzero()[0], 0, 0), gaps.size - 1)

        gap_spacings = [gaps[gap_chunks[0]]]
        for i_chunk in gap_chunks[1:-1]:
            gap_spacings.append(gaps[i_chunk + 1] - (gaps[i_chunk] + 1))
        gap_spacings.append(n_voxels - (gaps[-1] + 1))

        assertion = np.unique(gap_spacings).size == 1
        assert assertion, "Expected equal x gap spacings: {}, {}".format(gap_spacings, gaps)

        return gap_spacings[0]

    @staticmethod
    def _coo2nested(coo, coord_1, coord_2, coord_3, n_feats_in):
        coords_feats = {}
        for coord_1, coord_2, coord_3, coord_feat, feat in zip(
            coo.coords[coord_1], coo.coords[coord_2], coo.coords[coord_3], coo.data
        ):
            if coord_1 not in coords_feats:
                coords_feats[coord_1] = {}
                coords_feats[coord_1][coord_2] = {}
                coords_feats[coord_1][coord_2][coord_3] = [0] * n_feats_in
            elif coord_2 not in coords_feats[coord_1]:
                coords_feats[coord_1][coord_2] = {}
                coords_feats[coord_1][coord_2][coord_3] = [0] * n_feats_in
            elif coord_3 not in coords_feats[coord_1][coord_2]:
                coords_feats[coord_1][coord_2][coord_3] = [0] * n_feats_in
            coords_feats[coord_1][coord_2][coord_3][coord_feat] = feat

    """ End __init__ helpers """

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x_gaps, z_gaps = self._generate_random_mask()

        unmasked_coords, unmasked_adcs, masked_coords, masked_adcs = self._apply_mask(
            data["xyz"], x_gaps, z_gaps
        )

        if self.mask_type == MaskType.LOSS_ONLY:
            ret = self._getitem_mask_loss_only(
                unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps
            )
        elif self.mask_type == MaskType.REFLECTION:
            ret = self._getitem_mask_reflection(
                data["xyz"], data["zxy"],
                unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps
            )

        return ret

    """ __getitem__ helpers """

    def _getitem_mask_loss_only(
        self, unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
    ):
        input_coords, input_feats = [], []
        target_coords, target_feats = [], []
        for coord, feats in zip(unmasked_coords, unmasked_feats):
            input_coords.append(coord)
            input_feats.append(feats)
            target_coords.append(coord)
            target_feats.append(feats[0:self.n_feats_out])

        for coord, feats in zip(masked_coords, masked_feats):
            target_coords.append(coord)
            target_feats.append(feats[0:self.n_feats_out])

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats),
            "mask_x" : x_gaps,
            "mask_z" : z_gaps
        }

    def _getitem_mask_reflection(
        self, coordsxyz_feats, coordszxy_feats,
        unmasked_coords, unmasked_feats, masked_coords, masked_feats,
        x_gaps, z_gaps
    ):
        infill_coords = set()

        x_gaps_set, x_gap_pos_rflct_coord, x_gap_neg_rflct_coord = self._get_gap_reflect_coords(
            x_gaps
        )
        z_gaps_set, z_gap_pos_rflct_coord, z_gap_neg_rflct_coord = self._get_gap_reflect_coords(
            z_gaps
        )

        # Make reflections of tracks into x gaps
        for coord_x, coordsyz_feats in coordsxyz_feats.items():
            # In ROI for reflection in positive direction
            if coord_x + self.x_gap_size + 1 in x_gaps_set:
                reflect_x = x_gap_pos_rflct_coord[coord_x + self.x_gap_size + 1]

                # No packets at pixel next to gap
                if reflect_x not in coordsxyz_feats:
                    continue

                for coord_y, coordsz_feats in coordsxyz_feats[reflect_x].items():
                    for coord_z in coordsz_feats:
                        reflect_y, reflect_z = coord_y, coord_z

                        for coord_y, coordsz_feats in coordsyz_feats.items():
                            for coord_z in coordsz_feats:
                                infill_coords.add(
                                    (
                                        2 * reflect_x - coord_x,
                                        2 * reflect_y - coord_y,
                                        2 * reflect_z - coord_z
                                    )
                                )

            # Needs to be reflected in the negative x_direction
            elif coord_x - (self.x_gap_size + 1) in x_gaps_set:
                reflect_x = x_gap_neg_rflct_coord[coord_x - (self.x_gap_size + 1)]

                # No packets at pixel next to gap
                if reflect_x not in coordsxyz_feats:
                    continue

                for coord_y, coordsz_feats in coordsxyz_feats[reflect_x].items():
                    for coord_z in coordsz_feats:
                        reflect_y, reflect_z = coord_y, coord_z

                        for coord_y, coordsz_feats in coordsyz_feats.items():
                            for coord_z in coordsz_feats:
                                infill_coords.add(
                                    (
                                        2 * reflect_x - coord_x,
                                        2 * reflect_y - coord_y,
                                        2 * reflect_z - coord_z
                                    )
                                )

        # Make reflections of tracks into z gaps
        for coord_z, coordsxy_feats in coordszxy_feats.items():
            # In ROI for reflection in positive direction
            if coord_z + self.z_gap_size + 1 in z_gaps_set:
                # Packets might skip a z bin (pixels are waiting to self trigger) so allow wiggle
                reflect_z_max = z_gap_pos_rflct_coord[coord_z + self.z_gap_size + 1]
                for reflect_z in range(reflect_z_max - 3, reflect_z_max + 1):

                    # No packets at pixel next to gap
                    if reflect_z not in coordszxy_feats:
                        continue

                    for coord_x, coordsy_feats in coordszxy_feats[reflect_z].items():
                        for coord_y in coordsy_feats:
                            reflect_x, reflect_y = coord_x, coord_y

                            for coord_x, coordsy_feats in coordsxy_feats.items():
                                for coord_y in coordsy_feats:
                                    infill_coords.add(
                                        (
                                            2 * reflect_x - coord_x,
                                            2 * reflect_y - coord_y,
                                            2 * reflect_z - coord_z
                                        )
                                    )

            # In ROI for reflection in negative direction
            elif coord_z - (self.z_gap_size + 1) in z_gaps_set:
                # Packets might skip a z bin (pixels are waiting to self trigger) so allow wiggle
                reflect_z_min = z_gap_neg_rflct_coord[coord_z - (self.z_gap_size + 1)]
                for reflect_z in range(reflect_z_min, reflect_z_min + 4):

                    # No packets at pixel next to gap
                    if reflect_z not in coordszxy_feats:
                        continue

                    for coord_x, coordsy_feats in coordszxy_feats[reflect_z].items():
                        for coord_y in coordsy_feats:
                            reflect_x, reflect_y = coord_x, coord_y

                            for coord_x, coordsy_feats in coordsxy_feats.items():
                                for coord_y in coordsy_feats:
                                    infill_coords.add(
                                        (
                                            2 * reflect_x - coord_x,
                                            2 * reflect_y - coord_y,
                                            2 * reflect_z - coord_z
                                        )
                                    )

        # Smear signal mask in all directions
        signal_mask = set()
        for coord_x, coordsyz_feats in coordsxyz_feats.items():
            for coord_y, coordsz_feats in coordsyz_feats.items():
                for coord_z in coordsz_feats:
                    for shift_x in range(-2, 3):
                        for shift_y in range(-2, 3):
                            for shift_z in range(-5, 6):
                                signal_mask.add(
                                    (coord_x + shift_x, coord_y + shift_y, coord_z + shift_z)
                                )

        for coords in infill_coords:
            for shift_x in range(-2, 3):
                for shift_y in range(-2, 3):
                    for shift_z in range(-5, 6):
                        signal_mask.add(
                            (coords[0] + shift_x, coords[1] + shift_y, coords[2] + shift_z)
                        )

        input_coords, input_feats = [], []
        target_coords, target_feats = [], []
        for coord, feats in zip(unmasked_coords, unmasked_feats):
            input_coords.append(coord)
            input_feats.append(feats)
            target_coords.append(coord)
            target_feats.append(feats[0:self.n_feats_out])
            signal_mask.remove(tuple(coord))

        for coord, feats in zip(masked_coords, masked_feats):
            target_coords.append(coord)
            target_feats.append(feats[0:self.n_feats_out])
            signal_mask.remove(tuple(coord))

        for coord in signal_mask:
            coord = list(coord)
            input_coords.append(coord)
            input_feats.append([0, 0])
            target_coords.append(coord)
            target_feats.append([0])

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats)
        }

    def _generate_random_mask(self):
        x_gaps = (
            self.x_true_gaps +
            (
                np.random.choice([1, -1]) *
                np.random.randint(
                    self.x_true_gap_padding + self.x_gap_size,
                    self.x_gap_spacing - self.x_true_gap_padding - self.x_gap_size
                )
            )
        )
        z_gaps = (
            self.z_true_gaps +
            (
                np.random.choice([1, -1]) *
                np.random.randint(
                    self.z_true_gap_padding + self.z_gap_size,
                    self.z_gap_spacing - self.z_true_gap_padding - self.z_gap_size
                )
            )
        )

        return x_gaps, z_gaps

    @staticmethod
    def _apply_mask(coordsxyz_feats, x_gaps, z_gaps):
        x_gaps, z_gaps = set(x_gaps), set(z_gaps)

        masked_coords, masked_feats = [], []
        unmasked_coords, unmasked_feats = [], []

        for coord_x, coordsyz_feats in coordsxyz_feats.items():
            for coord_y, coordsz_feats in coordsyz_feats.items():
                for coord_z, feats in coordsz_feats.items():
                    if coord_x in x_gaps or coord_z in z_gaps:
                        masked_coords.append([coord_x, coord_y, coord_z])
                        masked_feats.append(feats)
                    else:
                        unmasked_coords.append([coord_x, coord_y, coord_z])
                        unmasked_feats.append(feats)

        return unmasked_coords, unmasked_feats, masked_coords, masked_feats

    @staticmethod
    def _get_gap_reflect_coords(gaps):
        gaps_set = set(gaps)

        gap_pos_rflct_coord, gap_neg_rflct_coord = {}, {}
        for gap_loc in gaps:
            gap_start = gap_loc
            while gap_start in gaps_set:
                gap_start -= 1
            gap_pos_rflct_coord[gap_loc] = gap_start + 1

            gap_end = gap_loc
            while gap_end in gaps_set:
                gap_end += 1
            gap_neg_rflct_coord[gap_loc] = gap_end - 1

        return gaps_set, gap_pos_rflct_coord, gap_neg_rflct_coord


    # tick_smear=60 is ~1cm each way
    def _reflect_into_x_gap(self, infill_coords, x_gaps, x_coords, x_gap_size, tick_smear=60):
        for gap_start in x_gaps[::2]:
            last_col = gap_start - 1
            if last_col not in x_coords:
                continue

            for coord in x_coords[last_col]:
                reflect_y, reflect_z = coord[1], coord[2]

                for col in range(gap_start - x_gap_size - 1, last_col):
                    if col not in x_coords:
                        continue

                    for coord in x_coords[col]:
                        reflected_xz_xy_yz = (
                            2 * last_col - coord[0],
                            2 * reflect_y - coord[1],
                            2 * reflect_z - coord[2]
                        )
                        for z_tick in range(-tick_smear, tick_smear + 1):
                            infill_coords.add(
                                (
                                    reflected_xz_xy_yz[0],
                                    reflected_xz_xy_yz[1],
                                    reflected_xz_xy_yz[2] + z_tick
                                )
                            )

        for gap_end in x_gaps[1::2]:
            first_col = gap_end + 1
            if first_col not in x_coords:
                continue

            for coord in x_coords[first_col]:
                reflect_y, reflect_z = coord[1], coord[2]

                for col in range(first_col + 1, gap_end + x_gap_size + 2):
                    if col not in x_coords:
                        continue

                    for coord in x_coords[col]:
                        reflected_xz_xy_yz = (
                            2 * first_col - coord[0],
                            2 * reflect_y - coord[1],
                            2 * reflect_z - coord[2]
                        )
                        for z_tick in range(-60, 61):
                            infill_coords.add(
                                (
                                    reflected_xz_xy_yz[0],
                                    reflected_xz_xy_yz[1],
                                    reflected_xz_xy_yz[2] + z_tick
                                )
                            )

    def _reflect_into_z_gap(self, infill_coords, z_gaps, z_coords, z_gap_size, tick_smear=60):
        for z_gap_start in z_gaps[::2]:
            for last_col_offset in range(1, z_gap_size):
                last_col = z_gap_start - last_col_offset
                if last_col not in z_coords:
                    continue

                for coord in z_coords[last_col]:
                    reflect_x, reflect_y = coord[0], coord[1]

                    for col in range(z_gap_start - z_gap_size - 1, last_col):
                        if col not in z_coords:
                            continue

                        for coord in z_coords[col]:
                            reflected_xz_xy_yz = (
                                2 * reflect_x - coord[0],
                                2 * reflect_y - coord[1],
                                # Still reflect into gap when starting further away from gap
                                2 * (z_gap_start - 1 - math.ceil(last_col_offset / 2)) - coord[2]
                            )
                            for z_tick in range(-tick_smear, tick_smear + 1):
                                infill_coords.add(
                                    (
                                        reflected_xz_xy_yz[0],
                                        reflected_xz_xy_yz[1],
                                        max(
                                            min(
                                                reflected_xz_xy_yz[2] + z_tick,
                                                z_gap_start + z_gap_size - 1
                                            ),
                                            z_gap_start
                                        )
                                    )
                                )

        for z_gap_end in z_gaps[1::2]:
            for first_col_offset in range(1, z_gap_size):
                first_col = z_gap_end + first_col_offset
                if first_col not in z_coords:
                    continue

                for coord in z_coords[first_col]:
                    reflect_x, reflect_y = coord[0], coord[1]

                    for col in range(first_col + 1, z_gap_end + z_gap_size + 2):
                        if col not in z_coords:
                            continue

                        for coord in z_coords[col]:
                            reflected_xz_xy_yz = (
                                2 * reflect_x - coord[0],
                                2 * reflect_y - coord[1],
                                2 * (z_gap_end + 1 + math.ceil(first_col_offset / 2)) - coord[2]
                            )
                            for z_tick in range(-tick_smear, tick_smear + 1):
                                infill_coords.add(
                                    (
                                        reflected_xz_xy_yz[0],
                                        reflected_xz_xy_yz[1],
                                        max(
                                            min(
                                                reflected_xz_xy_yz[2] + z_tick,
                                                z_gap_start + z_gap_size - 1
                                            ),
                                            z_gap_start
                                        )
                                    )
                                )

    """ End __getitem__ helpers """


class CollateCOO:
    def __init__(self, device):
        self.device = device

        self.required_keys = set(["input_coords", "input_feats", "target_coords", "target_feats"])

    def __call__(self, list_coo):
        list_input_coords = [ coo["input_coords"] for coo in list_coo ]
        list_input_feats = [ coo["input_feats"] for coo in list_coo ]
        list_target_coords = [ coo["target_coords"] for coo in list_coo ]
        list_target_feats = [ coo["target_feats"] for coo in list_coo ]

        input_coords, input_feats = ME.utils.sparse_collate(
            coords=list_input_coords, feats=list_input_feats
        )
        # NOTE Doing the next step and making a SparseTensor here causes problems when using
        # num_workers in DataLoader since subprocess needs tries to pickle the SparseTensor which
        # is not possible

        target_coords, target_feats = ME.utils.sparse_collate(
            coords=list_target_coords, feats=list_target_feats
        )

        ret = {
            "input_coords" : input_coords, "input_feats" : input_feats,
            target_coords : "target_coords", target_feats : "target_feats"
        }

        for extra_key in set(list_coo[0].keys()) - self.required_keys:
            ret[extra_key] = [ coo[extra_key] for coo in list_coo ]

        return ret


class MaskType(Enum):
    LOSS_ONLY = 1
    REFLECTION = 2


# Testing
if __name__ == "__main__":
    data_path = "/share/rcifdata/awilkins/larnd_infill_data/zdownsample10/all"
    vmap_path = "/home/awilkins/larnd_infill/larnd_infill/voxel_maps/vmap_zdownresolution10.yml"
    with open(vmap_path, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)
    dataset = LarndDataset(
        data_path, MaskType.REFLECTION, vmap, 2, 1, max_dataset_size=200, seed=1
    )
    device = torch.device("cuda:0")
    b_size = 4
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=b_size, collate_fn=CollateCOO(device), num_workers=4
    )
    dataloader_itr = iter(dataloader)
    s = time.time()
    num_iters = 20
    for i in range(num_iters):
        batch = next(dataloader_itr)
    e = time.time()
    print("Loaded {} images in {:.4f}s".format(b_size * num_iters, e - s))
    print(batch["input"].shape, batch["target"].shape)


