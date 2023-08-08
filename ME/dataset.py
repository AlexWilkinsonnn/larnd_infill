import os, time
from enum import Enum

import numpy as np
from tqdm import tqdm
import sparse, yaml

import torch
import MinkowskiEngine as ME

from aux import plot_ndlar_voxels_2


class LarndDataset(torch.utils.data.Dataset):
    """Dataset for reading sparse volexised larnd-sim data and preparing an infill mask"""
    def __init__(
        self, dataroot, prep_type, vmap, n_feats_in, n_feats_out, feat_scalefactors,
        x_true_gap_padding=None, z_true_gap_padding=None,
        valid=False, max_dataset_size=0, seed=None
    ):
        if seed is not None:
            np.random.seed(seed)

        self.prep_type = prep_type

        self.n_feats_in = n_feats_in
        # Assuming the first n features are the target features
        self.n_feats_out = n_feats_out

        self.feat_scalefactors = np.array(feat_scalefactors)
        assertion = self.feat_scalefactors.shape == (n_feats_in,)
        assert assertion, "Invalid scalefactors: shape {}".format(self.feat_scalefactors.shape)

        self.vmap = vmap
        self.x_true_gaps = np.array(vmap["x_gaps"])
        self.z_true_gaps = np.array(vmap["z_gaps"])
        self.max_x_voxel = vmap["n_voxels"]["x"] - 1
        self.max_y_voxel = vmap["n_voxels"]["y"] - 1
        self.max_z_voxel = vmap["n_voxels"]["z"] - 1

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

        # NOTE mutliprocessing does not speed up this loop
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

            data_path = os.path.join(data_dir, f)
            self.data[-1]["data_path"] = data_path

            coo = sparse.load_npz(data_path)

            # [x][y][z] = [feat_1, feat_2]
            self.data[-1]["xyz"] = self._coo2nested(coo, 0, 1, 2, n_feats_in)

            if prep_type == DataPrepType.REFLECTION:
                self.data[-1]["zxy"] = self._coo2nested(coo, 2, 0, 1, n_feats_in)

    """ __init__ helpers """

    @staticmethod
    def _calc_gap_size( gaps):
        gap_chunks = np.append(np.insert((np.diff(gaps) != 1).nonzero()[0], 0, 0), gaps.size - 1)

        gap_sizes = np.diff(gap_chunks)
        gap_sizes[0] += 1

        assertion = np.unique(gap_sizes).size == 1
        assert assertion, "Expected equal gap sizes: {}, {}".format(gap_sizes, gaps)

        return gap_sizes[0]

    @staticmethod
    def _calc_gap_spacing(gaps, n_voxels):
        gap_chunks = np.append(np.insert((np.diff(gaps) != 1).nonzero()[0], 0, 0), gaps.size - 1)

        gap_spacings = [gaps[gap_chunks[0]]]
        for i_chunk in gap_chunks[1:-1]:
            gap_spacings.append(gaps[i_chunk + 1] - (gaps[i_chunk] + 1))
        gap_spacings.append(n_voxels - (gaps[-1] + 1))

        assertion = np.unique(gap_spacings).size == 1
        assert assertion, "Expected equal gap spacings: {}, {}".format(gap_spacings, gaps)

        return gap_spacings[0]

    @staticmethod
    def _coo2nested(coo, coord_0, coord_1, coord_2, n_feats_in):
        coords_feats = {}
        for coord_0, coord_1, coord_2, coord_feat, feat in zip(
            coo.coords[coord_0], coo.coords[coord_1], coo.coords[coord_2], coo.coords[3], coo.data
        ):
            if coord_0 not in coords_feats:
                coords_feats[coord_0] = {}
                coords_feats[coord_0][coord_1] = {}
                coords_feats[coord_0][coord_1][coord_2] = [0] * n_feats_in
            elif coord_1 not in coords_feats[coord_0]:
                coords_feats[coord_0][coord_1] = {}
                coords_feats[coord_0][coord_1][coord_2] = [0] * n_feats_in
            elif coord_2 not in coords_feats[coord_0][coord_1]:
                coords_feats[coord_0][coord_1][coord_2] = [0] * n_feats_in

            coords_feats[coord_0][coord_1][coord_2][coord_feat] = feat

        return coords_feats

    """ End __init__ helpers """

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x_gaps, z_gaps = self._generate_random_mask()

        unmasked_coords, unmasked_feats, masked_coords, masked_feats = self._apply_mask(
            data["xyz"], x_gaps, z_gaps
        )

        # Normalise to [0,1]
        unmasked_feats *= self.feat_scalefactors
        masked_feats *= self.feat_scalefactors

        if self.prep_type == DataPrepType.STANDARD:
            ret = self._getitem_standard(
                unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
            )
        elif self.prep_type == DataPrepType.REFLECTION:
            ret = self._getitem_reflection(
                data["xyz"], data["zxy"],
                unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
            )

        elif self.prep_type == DataPrepType.GAP_DISTANCE:
            ret = self._getitem_gap_distance(
                unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
            )

        ret["data_path"] = data["data_path"]

        return ret

    """ __getitem__ helpers """

    def _getitem_standard(
        self, unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
    ):
        input_coords = unmasked_coords
        input_feats = unmasked_feats

        target_coords = np.concatenate((unmasked_coords, masked_coords))
        target_feats = np.concatenate((unmasked_feats, masked_feats))
        target_feats = target_feats[:, :self.n_feats_out]

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats),
            "mask_x" : x_gaps,
            "mask_z" : z_gaps
        }

    def _getitem_gap_distance(
        self, unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
    ):
        gap_distances_feats = np.zeros((unmasked_feats.shape[0], 2))

        gap_distances_feats[:, 0] = self._calc_gap_distance_feats(
            x_gaps, self.x_true_gaps, self.x_gap_spacing, unmasked_coords, 0
        )
        gap_distances_feats[:, 1] = self._calc_gap_distance_feats(
            z_gaps, self.z_true_gaps, self.z_gap_spacing, unmasked_coords, 2
        )

        input_coords = unmasked_coords
        input_feats = np.concatenate((unmasked_feats, gap_distances_feats), axis=1)

        target_coords = np.concatenate((unmasked_coords, masked_coords))
        target_feats = np.concatenate((unmasked_feats, masked_feats))
        target_feats = target_feats[:, :self.n_feats_out]

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats),
            "mask_x" : { gap for gap in x_gaps },
            "mask_z" : { gap for gap in z_gaps }
        }

    @staticmethod
    def _calc_gap_distance_feats(gaps, true_gaps, gap_spacing, unmasked_coords, coord_idx):
        gap_distances = np.zeros(unmasked_coords.shape[0])

        gap_borders = np.concatenate(
            tuple(
                chunk[::chunk.size-1]
                for chunk in np.split(gaps, np.where(np.diff(gaps) != 1)[0] + 1)
            )
        )
        gap_shift_from_true = gaps[0] - true_gaps[0]
        gap_starts = np.concatenate(([gap_shift_from_true - 1], gap_borders[1::2]))

        centre_offset = np.ceil(gap_spacing / 2)
        active_centres = gap_starts + centre_offset
        if not gap_spacing % 2: # Middle voxel is not unique so use both
            active_centres = np.concatenate((active_centres, gap_starts + centre_offset + 1))

        # Find the smallest distance from any of the possible centres
        dists_from_centre = unmasked_coords[:, [coord_idx]] - active_centres
        gap_distances = dists_from_centre[
            np.arange(unmasked_coords.shape[0]), np.abs(dists_from_centre).argmin(axis=1)
        ]
        gap_distances /= (centre_offset - 1) # Normalise to [-1, 1]

        # Distinguish between edges of detector and gaps
        gap_distances[unmasked_coords[:, coord_idx] < np.min(active_centres)] = 0.0
        gap_distances[unmasked_coords[:, coord_idx] > np.max(active_centres)] = 0.0

        return gap_distances

    def _getitem_reflection(
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
        self._reflect_into_x_gaps(
            coordsxyz_feats,
            x_gap_pos_rflct_coord, x_gap_neg_rflct_coord, x_gaps_set,
            infill_coords
        )

        # Make reflections of tracks into z gaps
        self._reflect_into_z_gaps(
            coordszxy_feats,
            z_gap_pos_rflct_coord, z_gap_neg_rflct_coord, z_gaps_set,
            3,
            infill_coords
        )

        # Smear signal mask in all directions
        signal_mask_active_coords, signal_mask_gap_coords = self._make_signal_mask(
            coordsxyz_feats, infill_coords,
            (-2, 3), (-2, 3), (-5, 6),
            (-1, 2), (-1, 2), (-3, 4),
            x_gaps_set, z_gaps_set
        )

        input_coords = unmasked_coords
        # Last entry of feat vec indicates whether coordinate is at gap or not
        input_feats = np.concatenate(
            (unmasked_feats, np.zeros((unmasked_coords.shape[0], 1))), axis=1
        )

        target_coords = np.concatenate((unmasked_coords, masked_coords))
        target_feats = np.concatenate((unmasked_feats, masked_feats))
        target_feats = target_feats[:, :self.n_feats_out]

        signal_mask_active_feats = torch.zeros(
            (signal_mask_active_coords.shape[0], self.n_feats_in + 1), dtype=torch.float
        )

        signal_mask_gap_feats = torch.zeros(
            (signal_mask_gap_coords.shape[0], self.n_feats_in + 1), dtype=torch.float
        )
        signal_mask_gap_feats[:, -1] = 1

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats),
            "signal_mask_active_coords" : torch.IntTensor(signal_mask_active_coords),
            "signal_mask_active_feats" : signal_mask_active_feats,
            "signal_mask_gap_coords" : torch.IntTensor(signal_mask_gap_coords),
            "signal_mask_gap_feats" : signal_mask_gap_feats,
            "mask_x" : x_gaps, "mask_z" : z_gaps
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

    def _apply_mask(self, coordsxyz_feats, x_gaps, z_gaps):
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

        masked_coords = self._list2array(masked_coords, int)
        masked_feats = self._list2array(masked_feats, float, coords=False)
        unmasked_coords = self._list2array(unmasked_coords, int)
        unmasked_feats = self._list2array(unmasked_feats, float, coords=False)

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

    def _reflect_into_x_gaps(
        self,
        coordsxyz_feats,
        x_gap_pos_rflct_coord, x_gap_neg_rflct_coord, x_gaps_set,
        infill_coords
    ):
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

    def _reflect_into_z_gaps(
        self,
        coordszxy_feats,
        z_gap_pos_rflct_coord, z_gap_neg_rflct_coord, z_gaps_set,
        wiggle,
        infill_coords
    ):
        for coord_z, coordsxy_feats in coordszxy_feats.items():
            # In ROI for reflection in positive direction
            if coord_z + self.z_gap_size + 1 in z_gaps_set:
                # Packets might skip a z bin (pixels are waiting to self trigger) so allow wiggle
                reflect_z_max = z_gap_pos_rflct_coord[coord_z + self.z_gap_size + 1]
                for reflect_z in range(reflect_z_max - wiggle, reflect_z_max + 1):

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
                for reflect_z in range(reflect_z_min, reflect_z_min + wiggle + 1):

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

    def _make_signal_mask(
        self, coordsxyz_feats, infill_coords,
        x_smear_infill, y_smear_infill, z_smear_infill,
        x_smear_active, y_smear_active, z_smear_active,
        x_gaps_set, z_gaps_set
    ):
        signal_mask_active, signal_mask_gap = set(), set()

        # NOTE Repeated coordinates will get ignored when making the sparse tensor anyway
        # but using sets here makes later operations faster

        for coords in infill_coords:
            for shift_x in range(*x_smear_infill):
                for shift_y in range(*y_smear_infill):
                    for shift_z in range(*z_smear_infill):
                        coord = (coords[0] + shift_x, coords[1] + shift_y, coords[2] + shift_z)
                        if coord[0] in x_gaps_set or coord[2] in z_gaps_set:
                            signal_mask_gap.add(coord)
                        else:
                            signal_mask_active.add(coord)

        for coord_x, coordsyz_feats in coordsxyz_feats.items():
            for coord_y, coordsz_feats in coordsyz_feats.items():
                for coord_z in coordsz_feats:
                    for shift_x in range(*x_smear_active):
                        for shift_y in range(*y_smear_active):
                            for shift_z in range(*z_smear_active):
                                coord = (coord_x + shift_x, coord_y + shift_y, coord_z + shift_z)
                                if coord[0] in x_gaps_set or coord[2] in z_gaps_set:
                                    signal_mask_gap.add(coord)
                                else:
                                    signal_mask_active.add(coord)

        signal_mask_active = self._list2array(
            [
                list(coord)
                for coord in signal_mask_active
                if (
                    coord[0] >= 0 and coord[0] <= self.max_x_voxel and
                    coord[1] >= 0 and coord[1] <= self.max_y_voxel and
                    coord[2] >= 0 and coord[2] <= self.max_z_voxel
                )
            ],
            int
        )
        signal_mask_gap = self._list2array(
            [
                list(coord)
                for coord in signal_mask_gap
                if (
                    coord[0] >= 0 and coord[0] <= self.max_x_voxel and
                    coord[1] >= 0 and coord[1] <= self.max_y_voxel and
                    coord[2] >= 0 and coord[2] <= self.max_z_voxel
                )
            ],
            int
        )

        return signal_mask_active, signal_mask_gap

    def _list2array(self, l, dtype, coords=True):
        return (
            np.array(l, dtype=dtype)
            if l else
            np.empty((0, 3) if coords else (0, self.n_feats_in), dtype=dtype)
        )

    """ End __getitem__ helpers """


class CollateCOO:
    def __init__(
        self, coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    ):
        self.coord_feat_pairs = coord_feat_pairs
        self.required_keys = { key for keys in coord_feat_pairs for key in keys }

    def __call__(self, data_list):
        ret = {}

        # Make batch list into a single tensor with a batch index in the coordinates
        for coords_key, feats_key in self.coord_feat_pairs:
            coords_list = [ data[coords_key] for data in data_list ]
            feats_list = [ data[feats_key] for data in data_list ]
            ret[coords_key], ret[feats_key] = ME.utils.sparse_collate(
                coords=coords_list, feats=feats_list
            )
            # NOTE Doing the next step and making a SparseTensor here causes problems when using
            # num_workers in DataLoader since subprocess needs tries to pickle the SparseTensor
            # which is not possible

        for extra_key in set(data_list[0].keys()) - self.required_keys:
            ret[extra_key] = [ coo[extra_key] for coo in data_list ]

        return ret


class DataPrepType(Enum):
    STANDARD = 1
    REFLECTION = 2
    GAP_DISTANCE = 3


# Testing
if __name__ == "__main__":
    data_path = "/share/rcifdata/awilkins/larnd_infill_data/zdownsample10/all"
    vmap_path = "/home/awilkins/larnd_infill/larnd_infill/voxel_maps/vmap_zdownresolution10.yml"
    with open(vmap_path, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)

    dataset = LarndDataset(
        data_path, DataPrepType.GAP_DISTANCE, vmap, 2, 1, [1 / 300, 1 / 5],
        max_dataset_size=200, seed=1
    )

    # collate_fn = CollateCOO(
    #     coord_feat_pairs=(
    #         ("input_coords", "input_feats"), ("target_coords", "target_feats"),
    #         ("signal_mask_active_coords", "signal_mask_active_feats"),
    #         ("signal_mask_gap_coords", "signal_mask_gap_feats")
    #     )
    # )
    collate_fn = CollateCOO(
        coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    )

    b_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=b_size, collate_fn=collate_fn, num_workers=0
    )

    dataloader_itr = iter(dataloader)
    s = time.time()
    num_iters = 5
    for i in range(num_iters):
        batch = next(dataloader_itr)
    e = time.time()

    print("Loaded {} images in {:.4f}s".format(b_size * num_iters, e - s))
    print(batch["input_coords"].shape, batch["input_feats"].shape)
    print(batch["target_coords"].shape, batch["target_feats"].shape)
    print(
        torch.unique(batch["input_coords"], dim=0).shape,
        torch.unique(batch["target_coords"], dim=0).shape
    )
    # print(batch["signal_mask_active_feats"])
    # print(batch["signal_mask_gap_feats"])
    # print(batch["signal_mask_active_coords"].shape, batch["signal_mask_active_feats"].shape)
    # print(batch["signal_mask_gap_coords"].shape, batch["signal_mask_gap_feats"].shape)

    from larpixsoft.detector import set_detector_properties
    det_props = "/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
    pixel_layout = (
        "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
    )
    detector = set_detector_properties(det_props, pixel_layout, pedestal=74)

    # coords_packed_sigmask_active = [[], [], []]
    # for coord in batch["signal_mask_active_coords"]:
    #     coords_packed_sigmask_active[0].append(coord[1].item())
    #     coords_packed_sigmask_active[1].append(coord[2].item())
    #     coords_packed_sigmask_active[2].append(coord[3].item())
    # coords_packed_sigmask_gap = [[], [], []]
    # for coord in batch["signal_mask_gap_coords"]:
    #     coords_packed_sigmask_gap[0].append(coord[1].item())
    #     coords_packed_sigmask_gap[1].append(coord[2].item())
    #     coords_packed_sigmask_gap[2].append(coord[3].item())

    coords_packed = [[], [], []]
    for coord in batch["input_coords"]:
        coords_packed[0].append(coord[1].item())
        coords_packed[1].append(coord[2].item())
        coords_packed[2].append(coord[3].item())
    coords_packed_target = [[], [], []]
    for coord in batch["target_coords"]:
        coords_packed_target[0].append(coord[1].item())
        coords_packed_target[1].append(coord[2].item())
        coords_packed_target[2].append(coord[3].item())

    # plot_ndlar_voxels_2(
    #     coords_packed, [ feats[0].item() for feats in batch["input_feats"] ],
    #     detector,
    #     vmap["x"], vmap["y"], vmap["z"],
    #     batch["mask_x"][0], batch["mask_z"][0],
    #     saveas=(
    #         "/home/awilkins/larnd_infill/larnd_infill/tests/input_sigmask_example_{}.pdf".format(
    #             os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
    #         )
    #     ),
    #     signal_mask_gap_coords=coords_packed_sigmask_gap,
    #     signal_mask_active_coords=coords_packed_sigmask_active,
    #     max_feat=1
    # )
    plot_ndlar_voxels_2(
        coords_packed, [ feats[0].item() for feats in batch["input_feats"] ],
        detector,
        vmap["x"], vmap["y"], vmap["z"],
        batch["mask_x"][0], batch["mask_z"][0],
        saveas=(
            "/home/awilkins/larnd_infill/larnd_infill/tests/input_adc_example{}_pretty.pdf".format(
                os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
            )
        ),
        max_feat=1, min_feat=-1,
        target_coords=coords_packed_target,
        target_feats=[ feats[0].item() for feats in batch["target_feats"] ],
        single_proj_pretty=True
    )
    # plot_ndlar_voxels_2(
    #     coords_packed, [ feats[2].item() for feats in batch["input_feats"] ],
    #     detector,
    #     vmap["x"], vmap["y"], vmap["z"],
    #     batch["mask_x"][0], batch["mask_z"][0],
    #     saveas=(
    #         "/home/awilkins/larnd_infill/larnd_infill/tests/input_xdistance_example{}.pdf".format(
    #             os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
    #         )
    #     ),
    #     max_feat=1, min_feat=-1
    # )
    # plot_ndlar_voxels_2(
    #     coords_packed, [ feats[3].item() for feats in batch["input_feats"] ],
    #     detector,
    #     vmap["x"], vmap["y"], vmap["z"],
    #     batch["mask_x"][0], batch["mask_z"][0],
    #     saveas=(
    #         "/home/awilkins/larnd_infill/larnd_infill/tests/input_zdistance_example{}.pdf".format(
    #             os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
    #         )
    #     ),
    #     max_feat=1, min_feat=-1
    # )

    # coords_packed = [[], [], []]
    # for coord in batch["target_coords"]:
    #     coords_packed[0].append(coord[1].item())
    #     coords_packed[1].append(coord[2].item())
    #     coords_packed[2].append(coord[3].item())
    # plot_ndlar_voxels_2(
    #     coords_packed, [ feats[0].item() for feats in batch["target_feats"] ],
    #     detector,
    #     vmap["x"], vmap["y"], vmap["z"],
    #     batch["mask_x"][0], batch["mask_z"][0],
    #     saveas=(
    #         "/home/awilkins/larnd_infill/larnd_infill/tests/target_sigmask_example_{}.pdf".format(
    #             os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
    #         )
    #     ),
    #     signal_mask_gap_coords=coords_packed_sigmask_gap,
    #     signal_mask_active_coords=coords_packed_sigmask_active,
    #     max_feat=1
    # )

