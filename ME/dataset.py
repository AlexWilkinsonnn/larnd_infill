import os, time
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
        self, dataroot, mask_type, vmap, n_feats_in, n_feats_out, feat_scalefactors,
        x_true_gap_padding=None, z_true_gap_padding=None,
        valid=False, max_dataset_size=0, seed=None

    ):
        if seed is not None:
            np.random.seed(seed)

        self.mask_type = mask_type

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

        if self.mask_type == MaskType.LOSS_ONLY:
            ret = self._getitem_mask_loss_only(
                unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
            )
        elif self.mask_type == MaskType.REFLECTION:
            ret = self._getitem_mask_reflection(
                data["xyz"], data["zxy"],
                unmasked_coords, unmasked_feats, masked_coords, masked_feats, x_gaps, z_gaps
            )

        ret["data_path"] = data["data_path"]

        return ret

    """ __getitem__ helpers """

    def _getitem_mask_loss_only(
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
        signal_mask_active, signal_mask_gap_input, signal_mask_gap_target = self._make_signal_mask(
            coordsxyz_feats,
            infill_coords,
            (-2, 3), (-2, 3), (-5, 6),
            x_gaps_set, z_gaps_set,
            { tuple(coord) for coord in unmasked_coords },
            { tuple(coord) for coord in masked_coords }
        )

        input_coords = np.concatenate((unmasked_coords, signal_mask_active, signal_mask_gap_input))
        input_feats = np.concatenate(
            (
                np.concatenate((unmasked_feats, np.zeros((unmasked_feats.shape[0], 1))), axis=1),
                np.zeros((signal_mask_active.shape[0] + signal_mask_gap_input.shape[0], 3))
            )
        )
        input_feats[-signal_mask_gap_input.shape[0]:, 2] = 1

        target_coords = np.concatenate(
            (unmasked_coords, masked_coords, signal_mask_active, signal_mask_gap_target)
        )
        target_feats = np.concatenate((unmasked_feats, masked_feats))
        target_feats = target_feats[:, :self.n_feats_out]
        target_feats = np.concatenate(
            (
                target_feats,
                np.zeros(
                    (
                        signal_mask_active.shape[0] + signal_mask_gap_target.shape[0],
                        self.n_feats_out
                    )
                )
            )
        )

        return {
            "input_coords" : torch.IntTensor(input_coords),
            "input_feats" : torch.FloatTensor(input_feats),
            "target_coords" : torch.IntTensor(target_coords),
            "target_feats" : torch.FloatTensor(target_feats),
            "mask_x" : x_gaps, "mask_z" : z_gaps
        }

#         input_coords, input_feats = [], []
#         target_coords, target_feats = [], []
#         for coord, feats in zip(unmasked_coords, unmasked_feats):
#             input_coords.append(coord)
#             input_feats.append(feats + [0])

#             target_coords.append(coord)
#             target_feats.append(feats[0:self.n_feats_out])

#             signal_mask.remove(tuple(coord))

#         masked_coords_set = set()
#         for coord, feats in zip(masked_coords, masked_feats):
#             target_coords.append(coord)
#             target_feats.append(feats[0:self.n_feats_out])

#             masked_coords_set.add(tuple(coord))

#         for coord in signal_mask:
#             if coord not in masked_coords_set:
#                 coord = list(coord)
#                 target_coords.append(coord)
#                 target_feats.append([0])
#             else:
#                 coord = list(coord)

#             input_coords.append(coord)
#             if coord[0] in x_gaps or coord[2] in z_gaps:
#                 input_feats.append([0, 0, 1])
#             else:
#                 input_feats.append([0, 0, 0])

#         return {
#             "input_coords" : torch.IntTensor(input_coords),
#             "input_feats" : torch.FloatTensor(input_feats),
#             "target_coords" : torch.IntTensor(target_coords),
#             "target_feats" : torch.FloatTensor(target_feats),
#             "mask_x" : x_gaps, "mask_z" : z_gaps
#         }

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
        self,
        coordsxyz_feats,
        infill_coords,
        x_smear, y_smear, z_smear,
        x_gaps_set, z_gaps_set,
        unmasked_coords_set, masked_coords_set
    ):
        signal_mask_active = set()
        signal_mask_gap_input, signal_mask_gap_target = set(), set()

        for coords in infill_coords:
            for shift_x in range(*x_smear):
                for shift_y in range(*y_smear):
                    for shift_z in range(*z_smear):
                        coord = (coords[0] + shift_x, coords[1] + shift_y, coords[2] + shift_z)
                        if (
                            coord[0] < 0 or coord[0] > self.max_x_voxel or
                            coord[1] < 0 or coord[1] > self.max_y_voxel or
                            coord[2] < 0 or coord[2] > self.max_z_voxel
                        ):
                            continue
                        if coord[0] in x_gaps_set or coord[2] in z_gaps_set:
                            signal_mask_gap_input.add(coord)
                            if coord not in masked_coords_set:
                                signal_mask_gap_target.add(coord)
                        else:
                            if coord not in unmasked_coords_set:
                                signal_mask_active.add(coord)

        for coord_x, coordsyz_feats in coordsxyz_feats.items():
            for coord_y, coordsz_feats in coordsyz_feats.items():
                for coord_z in coordsz_feats:
                    for shift_x in range(*x_smear):
                        for shift_y in range(*y_smear):
                            for shift_z in range(*z_smear):
                                coord = (coord_x + shift_x, coord_y + shift_y, coord_z + shift_z)
                                if (
                                    coord[0] < 0 or coord[0] > self.max_x_voxel or
                                    coord[1] < 0 or coord[1] > self.max_y_voxel or
                                    coord[2] < 0 or coord[2] > self.max_z_voxel
                                ):
                                    continue
                                if coord[0] in x_gaps_set or coord[2] in z_gaps_set:
                                    signal_mask_gap_input.add(coord)
                                    if coord not in masked_coords_set:
                                        signal_mask_gap_target.add(coord)
                                else:
                                    if coord not in unmasked_coords_set:
                                        signal_mask_active.add(coord)

        signal_mask_active = self._list2array(
            [ list(coord) for coord in signal_mask_active ], int
        )
        signal_mask_gap_input = self._list2array(
            [ list(coord) for coord in signal_mask_gap_input ], int
        )
        signal_mask_gap_target = self._list2array(
            [ list(coord) for coord in signal_mask_gap_target ], int
        )

        return signal_mask_active, signal_mask_gap_input, signal_mask_gap_target

    def _list2array(self, l, dtype, coords=True):
        return (
            np.array(l, dtype=dtype)
            if l else
            np.empty((0, 3) if coords else (0, self.n_feats_in), dtype=dtype)
        )

    """ End __getitem__ helpers """


class CollateCOO:
    def __init__(self):
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
            "target_coords" : target_coords, "target_feats" : target_feats
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
        data_path, MaskType.REFLECTION, vmap, 2, 1, [1 / 300, 1 / 5], max_dataset_size=200, seed=1
    )

    b_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=b_size, collate_fn=CollateCOO(), num_workers=0
    )

    dataloader_itr = iter(dataloader)
    s = time.time()
    num_iters = 3
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

    # import sys; sys.exit()

    from larpixsoft.detector import set_detector_properties
    det_props = "/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
    pixel_layout = (
        "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
    )
    detector = set_detector_properties(det_props, pixel_layout, pedestal=74)
    coords_packed = [[], [], []]
    for coord in batch["input_coords"]:
        coords_packed[0].append(coord[1].item())
        coords_packed[1].append(coord[2].item())
        coords_packed[2].append(coord[3].item())
    plot_ndlar_voxels_2(
        coords_packed, [ feats[0].item() for feats in batch["input_feats"] ],
        detector,
        vmap["x"], vmap["y"], vmap["z"],
        batch["mask_x"][0], batch["mask_z"][0],
        saveas=(
            "/home/awilkins/larnd_infill/larnd_infill/tests/input_sigmask_example_{}.pdf".format(
                os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
            )
        ),
        signal_mask=True, max_feat=1
    )
    coords_packed = [[], [], []]
    for coord in batch["target_coords"]:
        coords_packed[0].append(coord[1].item())
        coords_packed[1].append(coord[2].item())
        coords_packed[2].append(coord[3].item())
    plot_ndlar_voxels_2(
        coords_packed, [ feats[0].item() for feats in batch["target_feats"] ],
        detector,
        vmap["x"], vmap["y"], vmap["z"],
        batch["mask_x"][0], batch["mask_z"][0],
        saveas=(
            "/home/awilkins/larnd_infill/larnd_infill/tests/target_sigmask_example_{}.pdf".format(
                os.path.basename(".".join(batch["data_path"][0].split(".")[:-1]))
            )
        ),
        signal_mask=True, max_feat=1
    )

