import os, math, time
from collections import defaultdict
from enum import Enum

import numpy as np
import sparse

import torch
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties

from aux import plot_ndlar_voxels


class LarndDataset(torch.utils.data.Dataset):
    """Dataset for reading sparse volexised larnd-sim data and preparing an infill mask"""
    def __init__(
        self, dataroot, mask_type,
        x_true_gaps, x_gap_spacing, x_gap_size, x_gap_padding,
        z_true_gaps, z_gap_spacing, z_gap_size, z_gap_padding,
        valid=False, max_dataset_size=0, seed=None
    ):
        assert(x_gap_size + x_gap_padding < x_gap_spacing)
        assert(z_gap_size + z_gap_padding < z_gap_spacing)

        self.mask_type = mask_type

        # data_dir = os.path.join(dataroot, "valid" if valid else "train")
        data_dir = dataroot
        self.data, self.data_x_coords, self.data_z_coords = [], [], []
        for i, f in enumerate(os.listdir(data_dir)):
            if max_dataset_size and i >= max_dataset_size:
                break

            data = sparse.load_npz(os.path.join(data_dir, f))
            if len(data.coords[0]) > 1500:
                continue
            self.data.append(data)

            if mask_type == MaskType.REFLECTION:
                self.data_x_coords.append(defaultdict(list))
                self.data_z_coords.append(defaultdict(list))
                for coord_x, coord_y, coord_z in zip(*data.coords):
                    coord = (coord_x, coord_y, coord_z)
                    self.data_x_coords[-1][coord_x].append(coord)
                    self.data_z_coords[-1][coord_z].append(coord)

        self.x_true_gaps = x_true_gaps # array of x gap start and end voxels
        self.x_gap_spacing = x_gap_spacing
        self.x_gap_size = x_gap_size
        self.x_gap_padding = x_gap_padding

        self.z_true_gaps = z_true_gaps
        self.z_gap_spacing = z_gap_spacing
        self.z_gap_size = z_gap_size
        self.z_gap_padding = z_gap_padding

        if seed is not None:
            np.random.seed(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x_gaps, z_gaps = self._generate_random_mask()

        unmasked_coords, unmasked_adcs, masked_coords, masked_adcs = self._apply_mask(
            data.coords, data.data, x_gaps, z_gaps
        )

        if self.mask_type == MaskType.LOSS_ONLY:
            ret = self._getitem_mask_loss_only(
                unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps
            )
        elif self.mask_type == MaskType.REFLECTION:
            ret = self._getitem_mask_reflection(
                unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps, index
            )

        return ret

    def _getitem_mask_loss_only(
        self, unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps
    ):
        input_coords, input_feats = [], []
        target_coords, target_feats = [], []
        for coord, adc in zip(unmasked_coords, unmasked_adcs):
            input_coords.append(coord)
            input_feats.append([adc])
            target_coords.append(coord)
            target_feats.append([adc])

        for coord, adc in zip(masked_coords, masked_adcs):
            target_coords.append(coord)
            target_feats.append([adc])

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats),
            "mask_x" : x_gaps,
            "mask_z" : z_gaps
        }

    def _getitem_mask_reflection(
        self, unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps, index
    ):
        data_x_coords, data_z_coords = self.data_x_coords[index], self.data_z_coords[index]

        input_coords, input_feats = [], []
        target_coords, target_feats = [], []
        for coord, adc in zip(unmasked_coords, unmasked_adcs):
            input_coords.append(coord)
            input_feats.append([adc, 0])
            target_coords.append(coord)
            target_feats.append([adc])

        for coord, adc in zip(masked_coords, masked_adcs):
            target_coords.append(coord)
            target_feats.append([adc])

        infill_coords = set()
        self._reflect_into_x_gap(
            infill_coords, x_gaps, data_x_coords, self.x_gap_size, tick_smear=60
        )
        self._reflect_into_z_gap(
            infill_coords, z_gaps, data_z_coords, self.z_gap_size, tick_smear=60
        )

        for coord in infill_coords:
            input_coords.append(coord)
            input_feats.append([0, 1])

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
                    self.x_gap_padding + self.x_gap_size,
                    self.x_gap_spacing - self.x_gap_padding - self.x_gap_size
                )
            )
        )
        z_gaps = (
            self.z_true_gaps +
            (
                np.random.choice([1, -1]) *
                np.random.randint(
                    self.z_gap_padding + self.z_gap_size,
                    self.z_gap_spacing - self.z_gap_padding - self.z_gap_size
                )
            )
        )

        return x_gaps, z_gaps

    def _apply_mask(self, coords, adcs, x_gaps, z_gaps):
        unmasked_coords, unmasked_adcs = [], []
        masked_coords, masked_adcs = [], []
        for coord_x, coord_y, coord_z, adc in zip(*coords, adcs):
            if (
                any(0 <= coord_x - x_gap_start < self.x_gap_size for x_gap_start in x_gaps[::2]) or
                any(0 <= coord_z - z_gap_start < self.z_gap_size for z_gap_start in z_gaps[::2])
            ):
                masked_coords.append((coord_x, coord_y, coord_z))
                masked_adcs.append(adc) # masked data will be in target only
            else:
                unmasked_coords.append((coord_x, coord_y, coord_z))
                unmasked_adcs.append(adc)

        return unmasked_coords, unmasked_adcs, masked_coords, masked_adcs

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
        s_input = ME.SparseTensor(
            coordinates=input_coords, features=input_feats, device=self.device
        )

        target_coords, target_feats = ME.utils.sparse_collate(
            coords=list_target_coords, feats=list_target_feats
        )
        s_target = ME.SparseTensor(
            coordinates=target_coords, features=target_feats, device=self.device
        )

        ret = { "input" : s_input, "target" : s_target }

        for extra_key in set(list_coo[0].keys()) - self.required_keys:
            ret[extra_key] = [ coo[extra_key] for coo in list_coo ]

        return ret


class MaskType(Enum):
    LOSS_ONLY = 1
    REFLECTION = 2


# Testing
if __name__ == "__main__":
    path = "/share/rcifdata/awilkins/larnd_infill_data/all"
    PIXEL_COLS_PER_ANODE = 256
    PIXEL_COLS_PER_GAP = 11 # 4.14 / 0.38
    TICKS_PER_MODULE = 6117
    TICKS_PER_GAP = 79 # 1.3cm / (0.1us * 0.1648cm/us)
    x_gaps = []
    for i in range(5):
        x_gaps.append(PIXEL_COLS_PER_ANODE * (i + 1) + PIXEL_COLS_PER_GAP * i)
        x_gaps.append(PIXEL_COLS_PER_ANODE * (i + 1) + PIXEL_COLS_PER_GAP * (i + 1) - 1)
    z_gaps = []
    for i in range(7):
        z_gaps.append(TICKS_PER_MODULE * (i + 1) + TICKS_PER_GAP * i)
        z_gaps.append(TICKS_PER_MODULE * (i + 1) + TICKS_PER_GAP * (i + 1) - 1)
    dataset = LarndDataset(
        path, MaskType.LOSS_ONLY,
        x_gaps, PIXEL_COLS_PER_ANODE, PIXEL_COLS_PER_GAP, PIXEL_COLS_PER_GAP,
        z_gaps, TICKS_PER_MODULE, TICKS_PER_GAP, TICKS_PER_GAP,
        max_dataset_size=1000, seed=1
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=CollateCOO())
    dataloader_itr = iter(dataloader)
    batch = next(dataloader_itr)
    print(batch)
    print(batch["input"].shape, batch["target"].shape)

