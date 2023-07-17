import argparse, os, math

import sparse, h5py

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map
from larpixsoft.funcs import get_events_no_cuts

from aux import plot_ndlar_voxels

DET_PROPS="/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
PIXEL_LAYOUT=(
    "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
)
# DET_PROPS=(
#     "/home/alex/Documents/extrapolation/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
# )
# PIXEL_LAYOUT=(
#     "/home/alex/Documents/extrapolation/larnd-sim/larndsim/pixel_layouts/"
#     "multi_tile_layout-3.0.40.yaml"
# )


def main(args):
    detector = set_detector_properties(DET_PROPS, PIXEL_LAYOUT, pedestal=74)
    geometry = get_geom_map(PIXEL_LAYOUT)

    if detector.vdrift != 0.1596452482154287:
        raise ValueError("Expected vdrift=0.1596452482154287 got {}".format(detector.vdrift))

    PIXEL_COL_OFFSET = 128
    PIXEL_COLS_PER_ANODE = 256
    PIXEL_COLS_PER_GAP = 11 # 4.14 / 0.38
    PIXEL_ROWS_PER_ANODE = 800
    PIXEL_ROW_OFFSET = 405

    if args.z_downsample == 1:
        TICK_OFFSET = 0
        TICKS_PER_MODULE = 6314
        TICKS_PER_GAP = 81 # 1.3cm / (0.1us * 0.1596452482154287cm/us)
    elif args.z_downsample == 10:
        TICKS_PER_MODULE = 612
        TICKS_PER_GAP = 8
    else:
        raise NotImplementedError("z_downsample={}".format(args.z_downsample))


    f = h5py.File(args.input_file, "r")

    # print(detector.tpc_borders)
    # print(detector.pixel_pitch) # 0.38
    # print(detector.vdrift) # 0.1596452482154287
    # print(detector.time_sampling) # 0.1
    # print(detector.tpc_offsets)
    # return

    # xs, ys = [], []
    # ys_raw = []
    # for x, y in geometry.values():
    #     xs.append(math.floor(x / detector.pixel_pitch))
    #     ys.append(math.floor(y / detector.pixel_pitch))
    #     ys_raw.append(y - detector.pixel_pitch / 2)
    # print(min(xs), max(xs))
    # print(min(ys), max(ys))
    # print(min(ys_raw), max(ys_raw))
    # return
    # -128 127 (x)
    # -405 394 (y)
    # -154.0, 149.62 (y_raw)

    packets = get_events_no_cuts(
        f['packets'], f['mc_packets_assn'], f['tracks'], geometry, detector, no_tracks=True
    )

    tpc_offsets_x = sorted(list(set(offsets[0] for offsets in detector.tpc_offsets)))
    tpc_offsets_z = sorted(list(set(offsets[2] for offsets in detector.tpc_offsets)))

    for i_ev, event_packets in enumerate(packets):
        coords = [[], [], []]
        adcs = []
        num_before_trigger, num_zero_adc, num_high_z = 0, 0, 0
        voxel_zs = []
        for p in event_packets:
            # Very rarely some packets are one tick before the trigger packet,
            # not sure what causes this
            if p.timestamp < p.t_0:
                num_before_trigger += 1
                continue

            if p.ADC == 0:
                num_zero_adc += 1

            # This is either caused by longitudinal diffusion, or the interactions with the LAr
            # taking a non negligible amount of time ie. not being instantaneous with the t_0 flash
            # I dont think its the second one though, think the entire interaction is sub ns
            if p.z() > 50.4:
                num_high_z += 1
                continue

            voxel_x = math.floor(p.x / detector.pixel_pitch)
            voxel_x += PIXEL_COL_OFFSET
            voxel_x += (
                tpc_offsets_x.index(p.anode.tpc_x) * (PIXEL_COLS_PER_ANODE + PIXEL_COLS_PER_GAP)
            )
            coords[0].append(voxel_x)

            voxel_y = math.floor(p.y / detector.pixel_pitch)
            voxel_y += PIXEL_ROW_OFFSET
            coords[1].append(voxel_y)

            if p.io_group in [1, 2]:
                voxel_z = math.floor(
                    (p.z() / (detector.time_sampling * detector.vdrift)) /
                    (100.8 / TICKS_PER_MODULE)
                )
            else:
                voxel_z = math.floor(
                    ((100.8 - p.z()) / (detector.time_sampling * detector.vdrift)) /
                    (100.8 / TICKS_PER_MODULE)
                )
            voxel_zs.append(voxel_z)
            voxel_z += TICK_OFFSET
            voxel_z += tpc_offsets_z.index(p.anode.tpc_z) * (TICKS_PER_MODULE + TICKS_PER_GAP)
            coords[2].append(voxel_z)

            adcs.append(p.ADC)

        if args.plot_only:
            plot_ndlar_voxels(
                coords, adcs, detector,
                pix_cols_per_anode=PIXEL_COLS_PER_ANODE, pix_cols_per_gap=PIXEL_COLS_PER_GAP,
                pix_rows_per_anode=PIXEL_COLS_PER_ANODE,
                ticks_per_module=TICKS_PER_MODULE, ticks_per_gap=TICKS_PER_GAP
            )
            continue

        s_voxelised = sparse.COO(
            coords, adcs,
            shape=(
                (5 * PIXEL_COLS_PER_ANODE + 4 * PIXEL_COLS_PER_GAP),
                PIXEL_ROWS_PER_ANODE,
                (7 * TICKS_PER_MODULE + 6 * TICKS_PER_GAP)
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

