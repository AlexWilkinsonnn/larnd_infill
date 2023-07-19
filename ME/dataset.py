import os, math, time
from enum import Enum

import numpy as np
import sparse, yaml

import torch
import MinkowskiEngine as ME

from larpixsoft.detector import set_detector_properties

from aux import plot_ndlar_voxels


class LarndDataset(torch.utils.data.Dataset):
    """Dataset for reading sparse volexised larnd-sim data and preparing an infill mask"""
    def __init__(
        self, dataroot, mask_type, vmap,
        x_true_gap_padding=None, z_true_gap_padding=None,
        valid=False, max_dataset_size=0, seed=None

    ):
        self.mask_type = mask_type

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
        # self.data, self.data_x_coords, self.data_z_coords = [], [], []
        for i, f in enumerate(os.listdir(data_dir)):
            if max_dataset_size and i >= max_dataset_size:
                break

            data = sparse.load_npz(os.path.join(data_dir, f))
            # if len(data.coords[0]) > 1500:
            #     continue
            self.data.append(data)

            # if mask_type == MaskType.REFLECTION:
            #     self.data_x_coords.append(defaultdict(list))
            #     self.data_z_coords.append(defaultdict(list))
            #     for coord_x, coord_y, coord_z in zip(*data.coords):
            #         coord = (coord_x, coord_y, coord_z)
            #         self.data_x_coords[-1][coord_x].append(coord)
            #         self.data_z_coords[-1][coord_z].append(coord)



        if seed is not None:
            np.random.seed(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x_gaps, z_gaps = self._generate_random_mask()

        unmasked_coords, unmasked_adcs, masked_coords, masked_adcs = self._apply_mask(
            sparse.DOK.from_coo(data), x_gaps, z_gaps
        )

        if self.mask_type == MaskType.LOSS_ONLY:
            ret = self._getitem_mask_loss_only(
                unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps
            )
        elif self.mask_type == MaskType.REFLECTION:
            raise NotImplementedError("Need to change this with refactor")
            # ret = self._getitem_mask_reflection(
            #     unmasked_coords, unmasked_adcs, masked_coords, masked_adcs, x_gaps, z_gaps, index
            # )

        return ret

    def _calc_gap_size(self, gaps):
        gap_chunks = np.append(np.insert((np.diff(gaps) != 1).nonzero()[0], 0, 0), gaps.size - 1)

        gap_sizes = np.diff(gap_chunks)
        gap_sizes[0] += 1

        assertion = np.unique(gap_sizes).size == 1
        assert assertion, "Expected equal x gap sizes: {}, {}".format(gap_sizes, gaps)

        return gap_sizes[0]

    def _calc_gap_spacing(self, gaps, n_voxels):
        gap_chunks = np.append(np.insert((np.diff(gaps) != 1).nonzero()[0], 0, 0), gaps.size - 1)

        gap_spacings = [gaps[gap_chunks[0]]]
        for i_chunk in gap_chunks[1:-1]:
            gap_spacings.append(gaps[i_chunk + 1] - (gaps[i_chunk] + 1))
        gap_spacings.append(n_voxels - (gaps[-1] + 1))

        assertion = np.unique(gap_spacings).size == 1
        assert assertion, "Expected equal x gap spacings: {}, {}".format(gap_spacings, gaps)

        return gap_spacings[0]

    def _getitem_mask_loss_only(
        self, unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
    ):
        input_coords, input_feats = [], []
        target_coords, target_feats = [], []
        for coord, feats in zip(unmasked_coords, unmasked_feats):
            input_coords.append(coord)
            input_feats.append(feats)
            target_coords.append(coord)
            target_feats.append(feats[0:1]) # Target is adc only, not number of packets stacked

        for coord, feats in zip(masked_coords, masked_feats):
            target_coords.append(coord)
            target_feats.append(feats[0:1])

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

    def _apply_mask(self, dok, x_gaps, z_gaps):
        x_gaps, z_gaps = set(x_gaps), set(z_gaps)

        masked_coords, masked_feats = [], []
        unmasked_coords, unmasked_feats = [], []

        coords_data = dok.data
        coords = set(coords_data)
        num_feats = dok.shape[-1]
        while coords:
            coord = next(iter(coords))

            # dok must have an entry for each direction in feature space
            feats = []
            for i_feat in range(num_feats):
                coord = (*coord[:3], i_feat)
                feats.append(coords_data[coord])
                coords.remove(coord)

            coord_spatial = list(coord[:3])

            if coord_spatial[0] in x_gaps or coord_spatial[2] in z_gaps:
                masked_coords.append(coord_spatial)
                masked_feats.append(feats)
            else:
                unmasked_coords.append(coord_spatial)
                unmasked_feats.append(feats)

        return unmasked_coords, unmasked_feats, masked_coords, masked_feats

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
    data_path = "/share/rcifdata/awilkins/larnd_infill_data/zdownsample10/all"
    vmap_path = "/home/awilkins/larnd_infill/larnd_infill/voxel_maps/vmap_zdownresolution10.yml"
    with open(vmap_path, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)
    dataset = LarndDataset(data_path, MaskType.LOSS_ONLY, vmap, max_dataset_size=1000, seed=1)
    device = torch.device("cuda:0")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=CollateCOO(device))
    dataloader_itr = iter(dataloader)
    for i in range(5):
        s = time.time()
        batch = next(dataloader_itr)
        e = time.time()
        print("{:.4f}".format(e - s))
    print(batch)
    print(batch["input"].shape, batch["target"].shape)

