"""
Use statistice of real edep-simsim data to make edep-sim formatted events of straight lines in
the LAr.
"""
import argparse
from functools import partialmethod

import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from larpixsoft.detector import set_detector_properties

DETECTOR = set_detector_properties(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml",
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml",
    pedestal=74
)

segments_dtype = np.dtype([
    ("eventID", "u4"), ("trackID", "u4"), ("pdgId", "i4"),
    ("x_start", "f4"), ("x_end", "f4"), ("x", "f4"),
    ("y_start", "f4"), ("y_end", "f4"), ("y", "f4"),
    ("z_start", "f4"), ("z_end", "f4"), ("z", "f4"),
    ("t_start", "f4"), ("t_end", "f4"), ("t", "f4"),
    ("t0_start", "f4"), ("t0_end", "f4"), ("t0", "f4"),
    ("n_electrons", "u4"), ("n_photons", "u4"),
    ("tran_diff", "f4"), ("long_diff", "f4"),
    ("dx", "f4"), ("dEdx", "f4"), ("dE", "f4"),
    ("pixel_plane", "i4")
])

def main(args):
    # NOTE using edep sim coords where x is drift direction, this gets swapped with z in larndsim
    min_x, max_x = np.min(DETECTOR.tpc_borders[:,2,:]), np.max(DETECTOR.tpc_borders[:,2,:])
    min_y, max_y = np.min(DETECTOR.tpc_borders[:,1,:]), np.max(DETECTOR.tpc_borders[:,1,:])
    min_z, max_z = np.min(DETECTOR.tpc_borders[:,0,:]), np.max(DETECTOR.tpc_borders[:,0,:])

    with h5py.File(args.input_fname, "r") as f:
        trajectories = np.array(f["trajectories"])
        vertices = np.array(f["vertices"])
        segments = np.array(f["segments"])

    new_segments_list = []
    for ev_id in tqdm(np.unique(vertices["eventID"])):
        ev_segments = segments[segments["eventID"] == ev_id]

        # Try to ignore "cloud" of small depositions
        cut_off_dE = np.percentile(ev_segments["dE"], 1)
        ev_segments = ev_segments[ev_segments["dE"] > cut_off_dE]

        step_E = np.mean(ev_segments["dE"])
        step_size = np.mean(ev_segments["dx"])

        start_x, stop_x = np.random.uniform(min_x, max_x, 2)
        start_y, stop_y = np.random.uniform(min_y, max_y, 2)
        start_z, stop_z = np.random.uniform(min_z, max_z, 2)
        start_t0, stop_t0 = np.min(ev_segments["t0"]), np.max(ev_segments["t0"])

        if args.fix_x:
            stop_x = start_x
        if args.fix_y:
            stop_y = start_y
        if args.fix_z:
            stop_z = start_z

        dx = stop_x - start_x
        dy = stop_y - start_y
        dz = stop_z - start_z
        dt0 = stop_t0 - start_t0
        dr = np.sqrt(dx**2 + dy**2 + dz**2)
        num_steps = max(int(dr / step_size), 1)
        step_size = dr / num_steps

        x_eq = lambda t, dx=dx: start_x + dx * t
        y_eq = lambda t, dy=dy: start_y + dy * t
        z_eq = lambda t, dz=dz: start_z + dz * t
        t0_eq = lambda t, dt0=dt0: start_t0 + dt0 * t

        steps = np.arange(0, 1.0, 1 / num_steps)
        new_segments = np.empty(len(steps), dtype=segments_dtype)
        for i_seg, t_start in enumerate(steps):
            t_end = t_start + 1 / num_steps

            x_start, x_end = x_eq(t_start), x_eq(t_end)
            y_start, y_end = y_eq(t_start), y_eq(t_end)
            z_start, z_end = z_eq(t_start), z_eq(t_end)
            t0_start, t0_end = t0_eq(t_start), t0_eq(t_end)

            dx = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2 + (z_end - z_start)**2)

            new_segments[i_seg]["eventID"] = ev_segments["eventID"][0]
            new_segments[i_seg]["trackID"] = 0
            new_segments[i_seg]["pdgId"] = 0
            new_segments[i_seg]["x_start"] = x_start
            new_segments[i_seg]["x_end"] = x_end
            new_segments[i_seg]["x"] = (x_end + x_start) / 2
            new_segments[i_seg]["y_start"] = y_start
            new_segments[i_seg]["y_end"] = y_end
            new_segments[i_seg]["y"] = (y_end + y_start) / 2
            new_segments[i_seg]["z_start"] = z_start
            new_segments[i_seg]["z_end"] = z_end
            new_segments[i_seg]["z"] = (z_end + z_start) / 2
            new_segments[i_seg]["t_start"] = 0.0
            new_segments[i_seg]["t_end"] = 0.0
            new_segments[i_seg]["t"] = 0.0
            new_segments[i_seg]["t0_start"] = t0_start
            new_segments[i_seg]["t0_end"] = t0_end
            new_segments[i_seg]["t0"] = (t0_end + t0_start) / 2
            new_segments[i_seg]["n_electrons"] = 0
            new_segments[i_seg]["n_photons"] = 0
            new_segments[i_seg]["tran_diff"] = 0.0
            new_segments[i_seg]["long_diff"] = 0.0
            new_segments[i_seg]["dx"] = dx
            new_segments[i_seg]["dEdx"] = step_E / dx if dx > 0 else 0
            new_segments[i_seg]["dE"] = step_E
            new_segments[i_seg]["pixel_plane"] = 0

        new_segments_list.append(new_segments)

        if args.plot_only:
            plot_edepsim(new_segments)

    if not args.plot_only:
        with h5py.File(args.output_fname, "w") as f:
            f.create_dataset("trajectories", data=trajectories)
            f.create_dataset("vertices", data=vertices)
            f.create_dataset("segments", data=np.concatenate(new_segments_list, axis=0))

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_fname", type=str)
    parser.add_argument("output_fname", type=str)

    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--fix_x", action="store_true", help="start_x = stop_x (drift direction)")
    parser.add_argument("--fix_y", action="store_true", help="start_y = stop_y")
    parser.add_argument("--fix_z", action="store_true", help="start_z = stop_z (beam direction)")

    args = parser.parse_args()

    return args

def plot_edepsim(segments):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for s in segments:
        ax.plot(
            (s["x_start"], s["x_end"]), (s["y_start"], s["y_end"]), (s["z_start"], s["z_end"]),
            color="b"
        )
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()

    if args.batch_mode:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    main(args)

