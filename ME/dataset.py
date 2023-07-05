import os

import numpy as np
import sparse

import torch


class LarndDataset(torch.utils.data.Dataset):
    """Dataset for reading sparse volexised larnd-sim data and preparing an infill mask"""
    def __init__(
        self, dataroot,
        x_true_gaps, x_gap_spacing, x_gap_size, x_gap_padding,
        z_true_gaps, z_gap_spacing, z_gap_size, z_gap_padding,
        valid=False
    ):
        data_dir = os.path.join(dataroot, "valid" if valid else "train")
        self.data = [ sparse.load_npz(os.path.join(data_dir, f)) for f in os.listdir(data_dir) ]

        assert(x_gap_size + x_gap_padding < x_gap_spacing)
        assert(z_gap_size + z_gap_padding < z_gap_spacing)

        self.x_true_gaps = x_true_gaps
        self.x_gap_spacing = x_gap_spacing
        self.x_gap_size = x_gap_size
        self.x_gap_padding = x_gap_padding

        self.z_true_gaps = z_true_gaps
        self.z_gap_spacing = z_gap_spacing
        self.z_gap_size = z_gap_size
        self.z_gap_padding = z_gap_padding

    def __getitem__(self, index):
        event = self.data[index]

        x_infill_mask = np.roll(
            self.x_true_gaps,
            (
                np.random.choice([1, -1]) *
                np.random.randint(
                    self.x_gap_padding + self.x_gap_size,
                    self.x_gap_spacing - self.x_gap_padding - self.x_gap_size
                )
            )
        )
        z_infill_mask = np.roll(
            self.z_true_gaps,
            (
                np.random.choice([1, -1]) *
                np.random.randint(
                    self.z_gap_padding + self.z_gap_size,
                    self.z_gap_spacing - self.z_gap_padding - self.z_gap_size
                )
            )
        )

        coords = event.coords
        adcs = event.data
        features = []

        x_infill_mask_coords = set(x_infill_mask.nonzero()[0])
        z_infill_mask_coords = set(z_infill_mask.nonzero()[0])
        for coord, adc in zip(coords, adcs):
            if coord[0] in x_infill_mask_coords:
                pass








        


        

