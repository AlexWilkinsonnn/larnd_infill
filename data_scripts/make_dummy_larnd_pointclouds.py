"""
Make dataset of voxelised straight lines in ND-LAr
"""
import argparse, os
from functools import partialmethod

import sparse, yaml
import numpy as np
from tqdm import tqdm

from larpixsoft.detector import set_detector_properties
from aux import plot_ndlar_voxels_2

DETECTOR = set_detector_properties(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml",
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml",
    pedestal=74
)
STEPS_PER_VOXEL = 500
ADC_PER_VOXEL_STEP = 50


def main(args):
    with open(args.vmap, "r") as f:
        vmap = yaml.load(f, Loader=yaml.FullLoader)

    max_x = vmap["n_voxels"]["x"] - 1
    max_y = vmap["n_voxels"]["y"] - 1
    max_z = vmap["n_voxels"]["z"] - 1

    adc_per_step = ADC_PER_VOXEL_STEP / STEPS_PER_VOXEL

    for i_ev in tqdm(range(args.start_index, args.start_index + args.n)):
        start_x, stop_x = np.random.uniform(0, max_x, 2) if args.fix_x is None else args.fix_x
        start_y, stop_y = np.random.uniform(0, max_y, 2) if args.fix_y is None else args.fix_y
        start_z, stop_z = np.random.uniform(0, max_z, 2) if args.fix_z is None else args.fix_z

        dx = stop_x - start_x
        dy = stop_y - start_y
        dz = stop_z - start_z
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        num_steps = int(dr * STEPS_PER_VOXEL)

        x_eq = lambda t: start_x + dx * t
        y_eq = lambda t: start_y + dy * t
        z_eq = lambda t: start_z + dz * t

        # Step along 3d line and deposit adc evenly along voxels
        voxel_data = {}
        for t in np.linspace(0, 1.0, num_steps):
            coord = (int(x_eq(t)), int(y_eq(t)), int(z_eq(t)))
            if coord not in voxel_data:
                voxel_data[coord] = {}
                voxel_data[coord]["tot_adc"] = adc_per_step
            else:
                voxel_data[coord]["tot_adc"] += adc_per_step

        coords = [[], [], [], []]
        feats = []
        for coord, data in voxel_data.items():
            adc = data["tot_adc"]
            for i in range(3):
                coords[i].append(coord[i])
            coords[3].append(0)
            feats.append(adc)

        if args.plot_only:
            print(i_ev)
            plot_ndlar_voxels_2(
                [
                    [ coord for coord, coord_feat in zip(coords[i], coords[3]) if coord_feat == 0 ]
                    for i in range(3)
                ],
                [ feat for feat, coord_feat in zip(feats, coords[3]) if coord_feat == 0 ],
                DETECTOR,
                vmap["x"], vmap["y"], vmap["z"], vmap["x_gaps"], vmap["z_gaps"]
            )
            continue

        s_voxelised = sparse.COO(coords, feats, shape=(max_x, max_y, max_z, 1))

        sparse.save_npz(os.path.join(args.output_dir, "dummy_ev{}.npz".format(i_ev)), s_voxelised)

def parse_arguments():
    parser = argparse.ArgumentParser()

    def start_stop(s):
        tup = tuple(int(coord) for coord in s.split(","))
        if len(tup) == 2:
            return tup
        else:
            raise argparse.ArgumentTypeError("fix coordinate must be 'start,stop'")

    parser.add_argument("output_dir", type=str)
    parser.add_argument("vmap", type=str, help="Location of generated voxelisation map to use")
    parser.add_argument("n", type=int)

    parser.add_argument(
        "--plot_only", action="store_true"
    )
    parser.add_argument(
        "--batch_mode", action="store_true"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="starting number to use for naming output files"
    )
    parser.add_argument(
        "--fix_x", type=start_stop, help="fix x coordinate to 'start,stop'", default=None
    )
    parser.add_argument(
        "--fix_y", type=start_stop, help="fix y coordinate to 'start,stop'", default=None
    )
    parser.add_argument(
        "--fix_z", type=start_stop, help="fix z coordinate to 'start,stop'", default=None
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.batch_mode:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    main(args)

