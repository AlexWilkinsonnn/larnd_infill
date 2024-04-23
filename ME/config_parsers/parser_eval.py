import os, shutil
from collections import namedtuple

import yaml

from larpixsoft.detector import set_detector_properties
from larpixsoft.geometry import get_geom_map

from ME.dataset import DataPrepType

defaults = {
    "det_props" : (
        "/home/awilkins/larnd-sim/larnd-sim/larndsim/detector_properties/ndlar-module.yaml"
    ),
    "pixel_layout" : (
        "/home/awilkins/larnd-sim/larnd-sim/larndsim/pixel_layouts/multi_tile_layout-3.0.40.yaml"
    ),
    "device" : "cuda:0",
    "adc_threshold" : 1,
    "xyz_smear_infill" : ((-1, 2), (-1, 2), (-3, 4)),
    "xyz_smear_active" : ((0, 1), (0, 1), (0, 1)),
    "model_params" : {},
    "load_G" : "",
    "output_file" : "out.h5",
    "smear_z" : 1
}

mandatory_fields = {
    "vmap_path",
    "input_file",
    "data_prep_type",
    "scalefactors",
    "n_feats_in", "n_feats_out",
    "batch_size",
    "cache_dir"
}

def get_config(conf_file, overwrite_dict={}):
    print("Reading conf from {}".format(conf_file))

    with open(conf_file) as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)

    for field, val in overwrite_dict.items():
        conf_dict[field] = val

    missing_fields = mandatory_fields - set(conf_dict.keys())
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in conf file at {}".format(missing_fields, conf_file)
        )

    for option in set(defaults.keys()) - set(conf_dict.keys()):
        conf_dict[option] = defaults[option]

    conf_dict["detector"] = set_detector_properties(
        conf_dict["det_props"], conf_dict["pixel_layout"], pedestal=74
    )
    conf_dict["geometry"] = get_geom_map(conf_dict["pixel_layout"])
    del conf_dict["det_props"]
    del conf_dict["pixel_layout"]

    with open(conf_dict["vmap_path"], "r") as f:
        conf_dict["vmap"] = yaml.load(f, Loader=yaml.FullLoader)
    del conf_dict["vmap_path"]

    if conf_dict["data_prep_type"] == "standard":
        conf_dict["data_prep_type"] = DataPrepType.STANDARD
    elif conf_dict["data_prep_type"] == "reflection":
        conf_dict["data_prep_type"] = DataPrepType.REFLECTION
    elif conf_dict["data_prep_type"] == "reflection_separate_masks":
        conf_dict["data_prep_type"] = DataPrepType.REFLECTION_SEPARATE_MASKS
    elif conf_dict["data_prep_type"] == "reflection_norandom":
        conf_dict["data_prep_type"] = DataPrepType.REFLECTION_NORANDOM
        if conf_dict["refresh_masks_epoch"] > 1:
            print(
                "data_prep_type 'reflection_norandom' already caches. " +
                "Setting refresh_masks_epoch to 1"
            )
            conf_dict["refresh_masks_epoch"] = 1
    elif conf_dict["data_prep_type"] == "gap_distance":
        conf_dict["data_prep_type"] = DataPrepType.GAP_DISTANCE
    else:
        raise ValueError("data_prep_type={} not recognised".format(conf_dict["data_prep_type"]))

    if conf_dict["adc_threshold"] is not None:
        conf_dict["adc_threshold"] = conf_dict["adc_threshold"] * conf_dict["scalefactors"][0]

    conf_namedtuple = namedtuple("conf", conf_dict)
    conf = conf_namedtuple(**conf_dict)

    return conf

