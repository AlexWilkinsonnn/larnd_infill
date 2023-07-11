import argparse, os, math

import sparse, h5py

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

from aux import plot_ndlar_voxels, plot_ndlar_voxels_2

# DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
# PIXEL_LAYOUT=(
#     "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
# )
DET_PROPS=(
    "/home/alex/Documents/extrapolation/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
)
PIXEL_LAYOUT=(
    "/home/alex/Documents/extrapolation/larnd-sim/larndsim/pixel_layouts/"
    "multi_tile_layout-3.0.40.yaml"
)
X_GAP_SIZE = 4.14 # cm
PIXEL_COL_OFFSET = 128
PIXEL_COLS_PER_ANODE = 256
PIXEL_COLS_PER_X_GAP = 11 # 4.14 / 0.38
PIXEL_ROWS_PER_ANODE = 800
PIXEL_ROW_OFFSET = 405
Z_MODULE_SIZE = 100.8 # cm
Z_GAP_SIZE = 1.3 # cm
TICK_OFFSET = 0
TICKS_PER_MODULE = 6117
TICKS_PER_Z_GAP = 79 # 1.3cm / (0.1us * 0.1648cm/us)


def main(args):
    if args.z_downsample == 1:
        TICKS_PER_MODULE = 6117
        TICKS_PER_Z_GAP = 79 # 1.3cm / (0.1us * 0.1648cm/us)
    elif args.z_downsample == 10:
        TICKS_PER_MODULE = 612
        TICKS_PER_Z_GAP = 8
    else:
        raise NotImplementedError("z_downsample = {}".format(args.z_downsample))

    module_pixel_size = 0.38
    module_tick_size = Z_MODULE_SIZE / TICKS_PER_MODULE
    gap_pixel_size = X_GAP_SIZE / PIXEL_COLS_PER_X_GAP
    gap_tick_size = Z_GAP_SIZE / TICKS_PER_Z_GAP

    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    f = h5py.File(args.input_file, "r")

    # print(detector.tpc_borders)
    # print(detector.pixel_pitch)
    # print(detector.vdrift)
    # print(detector.time_sampling)
    # print((356.7 - 255.9) / (0.1 * 0.1648))
    # print(detector.tpc_offsets)
    # import sys; sys.exit()

    # xs, ys = [], []
    # for x, y in geometry.values():
    #     xs.append(math.floor(x / detector.pixel_pitch))
    #     ys.append(math.floor(y / detector.pixel_pitch))
    # print(min(xs), max(xs))
    # print(min(ys), max(ys))
    # import sys; sys.exit()
    # -128 127 (x)
    # -405 394 (y)

    packets = get_events_no_cuts(
        f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=True
    )

    tpc_offsets_x = sorted(list(set(offsets[0] for offsets in detector.tpc_offsets)))
    tpc_offsets_z = sorted(list(set(offsets[2] for offsets in detector.tpc_offsets)))

    for i_ev, event_packets in enumerate(packets):
        coords = [[], [], []]
        adcs = []
        num_before_trigger, num_zero_adc, num_high_z = 0, 0, 0
        for p in event_packets:
            # Very rarely some packets are one tick before the trigger packet,
            # not sure what causes this
            if p.timestamp < p.t_0:
                num_before_trigger += 1
                continue

            if p.ADC == 0:
                num_zero_adc += 1
                continue

            # This is either caused by longitudinal diffusion, or the interactions with the LAr
            # taking a non negligible amount of time ie. not being instantaneous with the t_0 flash
            # I dont think its the second one though, think the entire interaction is sub ns
            if p.z() > 50.4:
                num_high_z += 1
                continue

            voxel_x = math.floor(p.x / detector.pixel_pitch)
            voxel_x += PIXEL_COL_OFFSET
            voxel_x += (
                tpc_offsets_x.index(p.anode.tpc_x) * (PIXEL_COLS_PER_ANODE + PIXEL_COLS_PER_X_GAP)
            )
            coords[0].append(voxel_x)

            voxel_y = math.floor(p.y / detector.pixel_pitch)
            voxel_y += PIXEL_ROW_OFFSET
            coords[1].append(voxel_y)

            if p.io_group in [1, 2]:
                voxel_z = math.floor(p.z() / (100.8 / TICKS_PER_MODULE))
            else:
                voxel_z = math.floor((100.8 - p.z()) / (100.8 / TICKS_PER_MODULE))
            voxel_z += TICK_OFFSET
            voxel_z += tpc_offsets_z.index(p.anode.tpc_z) * (TICKS_PER_MODULE + TICKS_PER_Z_GAP)
            coords[2].append(voxel_z)

            adcs.append(p.ADC)

        if args.plot_only:
            x_vox2pos, x_vox2size = {}, {}
            x_pos, x_vox, next_gap_start = 0.0, 0, PIXEL_COLS_PER_ANODE
            while x_vox < 5 * PIXEL_COLS_PER_ANODE + 4 * PIXEL_COLS_PER_X_GAP:
                if x_vox == next_gap_start:
                    next_gap_start += PIXEL_COLS_PER_X_GAP + PIXEL_COLS_PER_ANODE
                    for _ in range(PIXEL_COLS_PER_X_GAP):
                        x_vox2pos[x_vox] = x_pos
                        x_vox2size[x_vox] = gap_pixel_size
                        x_pos += gap_pixel_size
                        x_vox += 1
                    continue
                x_vox2pos[x_vox] = x_pos
                x_vox2size[x_vox] = module_pixel_size
                x_pos += module_pixel_size
                x_vox += 1

            z_vox2pos, z_vox2size = {}, {}
            z_pos, z_vox, next_gap_start = 0.0, 0, TICKS_PER_MODULE
            while z_vox < 7 * TICKS_PER_MODULE + 6 * TICKS_PER_Z_GAP:
                if z_vox == next_gap_start:
                    next_gap_start += TICKS_PER_Z_GAP + TICKS_PER_MODULE
                    for _ in range(TICKS_PER_Z_GAP):
                        z_vox2pos[z_vox] = z_pos
                        z_vox2size[z_vox] = gap_tick_size
                        z_pos += gap_tick_size
                        z_vox += 1
                    continue
                z_vox2pos[z_vox] = z_pos
                z_vox2size[z_vox] = module_tick_size
                z_pos += module_tick_size
                z_vox += 1

            y_vox2pos, y_vox2size = {}, {}
            y_pos, y_vox = 0.0, 0
            while y_vox < PIXEL_ROWS_PER_ANODE:
                y_vox2pos[y_vox] = y_pos
                y_vox2size[y_vox] = module_pixel_size
                y_pos += module_pixel_size
                y_vox += 1

            print(num_before_trigger)
            print(num_high_z)
            print()
            plot_ndlar_voxels_2(
                coords, adcs, detector,
                x_vox2pos, y_vox2pos, z_vox2pos, x_vox2size, y_vox2size, z_vox2size,
                z_scalefactor=5
            )
            # plot_ndlar_voxels(
            #     coords, adcs, detector,
            #     pix_cols_per_anode=PIXEL_COLS_PER_ANODE, pix_cols_per_gap=PIXEL_COLS_PER_X_GAP,
            #     pix_rows_per_anode=PIXEL_COLS_PER_ANODE,
            #     ticks_per_module=TICKS_PER_MODULE, ticks_per_gap=TICKS_PER_Z_GAP,
            #     projections=True, structure=False, z_downsample=args.z_downsample
            # )
            continue

        s_voxelised = sparse.COO(
            coords, adcs,
            shape=(
                (5 * PIXEL_COLS_PER_ANODE + 4 * PIXEL_COLS_PER_X_GAP),
                PIXEL_ROWS_PER_ANODE,
                (7 * TICKS_PER_MODULE + 6 * TICKS_PER_Z_GAP)
            )
        )

        sparse.save_npz(
            os.path.join(
                args.output_dir,
                os.path.basename(args.input_file).split(".h5")[0] + "_ev{}.npz".format(i_ev)
            ),
            s_voxelised
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_dir")

    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--z_downsample", type=int, default=1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

