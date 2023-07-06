import os
from collections import defaultdict

import numpy as np
import sparse
from tqdm import tqdm

import torch


class LarndDataset(torch.utils.data.Dataset):
    """Dataset for reading sparse volexised larnd-sim data and preparing an infill mask"""
    def __init__(
        self, dataroot,
        x_true_gaps, x_gap_spacing, x_gap_size, x_gap_padding,
        z_true_gaps, z_gap_spacing, z_gap_size, z_gap_padding,
        valid=False
    ):
        # data_dir = os.path.join(dataroot, "valid" if valid else "train")
        data_dir = dataroot
        self.data = []
        for i, f in enumerate(os.listdir(data_dir)):
            self.data.append(sparse.load_npz(os.path.join(data_dir, f)))
            if i > 100:
                break
                      
        # self.data = [ sparse.load_npz(os.path.join(data_dir, f)) for f in os.listdir(data_dir) ]

        assert(x_gap_size + x_gap_padding < x_gap_spacing)
        assert(z_gap_size + z_gap_padding < z_gap_spacing)

        self.x_true_gaps = x_true_gaps # array of x gap start and end voxels
        self.x_gap_spacing = x_gap_spacing
        self.x_gap_size = x_gap_size
        self.x_gap_padding = x_gap_padding

        self.z_true_gaps = z_true_gaps
        self.z_gap_spacing = z_gap_spacing
        self.z_gap_size = z_gap_size
        self.z_gap_padding = z_gap_padding

    def __getitem__(self, index):
        event = self.data[index]

        adc_coords = event.coords
        adcs = event.data

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

        infill_coords = set()
        x_coords_near_gap = defaultdict()
        z_coords_near_gap = defaultdict()
        for coord_x, coord_y, coord_z, adc in zip(*adc_coords, adcs):
            if any(0 < x_gap_start - coord_x <= self.x_gap_size for x_gap_start in x_gaps[::2]):
                x_coords_near_gap[coord_x] = (coord_x, coord_y, coord_z, adc)
            elif any(0 < coord_x - x_gap_end <= self.x_gap_size for x_gap_end in x_gaps[1::2]):
                x_coords_near_gap[coord_x] = (coord_x, coord_y, coord_z, adc)
            if any(0 < z_gap_start - coord_z <= self.z_gap_size for z_gap_start in z_gaps[::2]):
                z_coords_near_gap[coord_z] = (coord_z, coord_y, coord_z, adc)
            elif any(0 < coord_z - z_gap_end <= self.z_gap_size for z_gap_end in z_gaps[1::2]):
                z_coords_near_gap[coord_z] = (coord_z, coord_y, coord_z, adc)

        print(x_gaps)
        print(z_gaps)
        print(x_coords_near_gap)
        print(z_coords_near_gap)
        return

        # Don't apply infill mask where there are no tracks
        xs, ys, zs = [], [], []
        for coord_x, coord_y, coord_z in zip(adc_coords[0], adc_coords[1], adc_coords[2]):
            xs.append(coord_x)
            ys.append(coord_y)
            zs.append(coord_z)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        x_infill_mask = np.diff(
            np.roll(
                self.x_true_gaps,
                (
                    np.random.choice([1, -1]) *
                    np.random.randint(
                        self.x_gap_padding + self.x_gap_size,
                        self.x_gap_spacing - self.x_gap_padding - self.x_gap_size
                    )
                )
            )
        )
        z_infill_mask = np.diff(
            np.roll(
                self.z_true_gaps,
                (
                    np.random.choice([1, -1]) *
                    np.random.randint(
                        self.z_gap_padding + self.z_gap_size,
                        self.z_gap_spacing - self.z_gap_padding - self.z_gap_size
                    )
                )
            )
        )

        features = []

        x_infill_mask_coords = np.concatenate(
            (np.where(x_infill_mask == 1)[0] + 1, np.where(x_infill_mask == -1)[0])
        )
        z_infill_mask_coords = np.concatenate(
            (np.where(z_infill_mask == 1)[0] + 1, np.where(z_infill_mask == -1)[0])
        )

        infill_mask_coords = set()
        for x in x_infill_mask_coords:
            if x < min_x or x > max_x:
                continue
            for z in range(min_z, max_z):
                for y in range(min_y, max_y):
                    infill_mask_coords.add((x,y,z))

        for z in z_infill_mask_coords:
            if z < min_z or z > max_z:
                continue
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    infill_mask_coords.add((x,y,z))

        print(len(infill_mask_coords))
        return

        for x in x_infill_mask_coords:
            for z in z_infill_mask_coords:
                for y in range(min_y, max_y):
                    infill_mask_coords.add((x,y,z))

        coords = []
        for i in range(len(adcs)):
            adc_coord, adc = (adc_coords[0][i], adc_coords[1][i], adc_coords[2][i]), adcs[i]
            coords.append(list(adc_coord))
            if adc_coord in x_infill_mask_coords:
                features.append([adc, 1])
                infill_mask_coords.remove(adc_coord)
            else:
                features.append([adc, 0])

        while infill_mask_coords:
            infill_coord = list(infill_mask_coords.pop())
            coords.append(infill_coord)
            features.append([0, 1])

        print(len(coords), len(features))
        

# Testing
if __name__ == "__main__":
    # path = "/share/lustre/awilkins/larnd_infill_data/all/"
    path = "/home/alex/Documents/extrapolation/larnd_infill_test_data"
    PIXEL_COLS_PER_ANODE = 256
    PIXEL_COLS_PER_GAP = 11 # 4.14 / 0.38
    TICKS_PER_MODULE = 6117
    TICKS_PER_GAP = 79 # 1.3cm / (0.1us * 0.1648cm/us)
    # x_gaps = np.zeros(PIXEL_COLS_PER_ANODE * 5 + PIXEL_COLS_PER_GAP * 4)
    # x_gap_starts = [ (i + 1) * PIXEL_COLS_PER_ANODE + i * PIXEL_COLS_PER_GAP for i in range(5) ]
    # for start in x_gap_starts:
    #     x_gaps[start:start+PIXEL_COLS_PER_GAP] = 1
    # z_gaps = np.zeros(TICKS_PER_MODULE * 7 + TICKS_PER_GAP * 6)
    # z_gap_starts = [ (i + 1) * TICKS_PER_MODULE + i * TICKS_PER_GAP for i in range(7) ]
    # for start in z_gap_starts:
    #     z_gaps[start:start+TICKS_PER_GAP] = 1
    x_gaps = []
    for i in range(5):
        x_gaps.append(PIXEL_COLS_PER_ANODE * (i + 1))
        x_gaps.append(PIXEL_COLS_PER_ANODE * (i + 1) + PIXEL_COLS_PER_GAP - 1)
    z_gaps = []
    for i in range(5):
        z_gaps.append(PIXEL_COLS_PER_ANODE * (i + 1))
        z_gaps.append(PIXEL_COLS_PER_ANODE * (i + 1) + PIXEL_COLS_PER_GAP - 1)
    dataset = LarndDataset(
        path,
        x_gaps, PIXEL_COLS_PER_ANODE, PIXEL_COLS_PER_GAP, PIXEL_COLS_PER_GAP,
        z_gaps, TICKS_PER_MODULE, TICKS_PER_GAP, TICKS_PER_GAP
    )
    dataset_itr = iter(dataset)
    next(dataset_itr)
    next(dataset_itr)
    next(dataset_itr)
    next(dataset_itr)

