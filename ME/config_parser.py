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
    "max_num_workers" : 4,
    "adc_threshold" : 1,
    "xyz_smear_infill" : ((-1, 2), (-1, 2), (-3, 4)),
    "xyz_smear_active" : ((0, 1), (0, 1), (0, 1)),
    "model_params" : {}
}

mandatory_fields = {
    "vmap_path", "data_path",
    "data_prep_type",
    "scalefactors",
    "n_feats_in", "n_feats_out",
    "max_dataset_size",
    "max_valid_dataset_size",
    "batch_size",
    "initial_lr",
    "loss_func",
    "epochs",
    "lr_decay_iter",
    "checkpoints_dir",
    "name"
}


def get_config(config_file, overwrite_dict={}, prep_checkpoint_dir=True):
    print("Reading config from {}".format(config_file))

    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    for field, val in overwrite_dict.items():
        config_dict[field] = val

    missing_fields = mandatory_fields - set(config_dict.keys())
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in config file at {}".format(missing_fields, config_file)
        )

    for option in set(defaults.keys()) - set(config_dict.keys()):
        config_dict[option] = defaults[option]

    config_dict["detector"] = set_detector_properties(
        config_dict["det_props"], config_dict["pixel_layout"], pedestal=74
    )
    config_dict["geometry"] = get_geom_map(config_dict["pixel_layout"])
    del config_dict["det_props"]
    del config_dict["pixel_layout"]

    with open(config_dict["vmap_path"], "r") as f:
        config_dict["vmap"] = yaml.load(f, Loader=yaml.FullLoader)
    del config_dict["vmap_path"]

    if config_dict["data_prep_type"] == "standard":
        config_dict["data_prep_type"] = DataPrepType.STANDARD
    elif config_dict["data_prep_type"] == "reflection":
        config_dict["data_prep_type"] = DataPrepType.REFLECTION
    elif config_dict["data_prep_type"] == "reflection_separate_masks":
        config_dict["data_prep_type"] = DataPrepType.REFLECTION_SEPARATE_MASKS
    elif config_dict["data_prep_type"] == "reflection_norandom":
        config_dict["data_prep_type"] = DataPrepType.REFLECTION_NORANDOM
    elif config_dict["data_prep_type"] == "gap_distance":
        config_dict["data_prep_type"] = DataPrepType.GAP_DISTANCE
    else:
        raise ValueError("data_prep_type={} not recognised".format(config_dict["data_prep_type"]))

    config_dict["train_data_path"] = os.path.join(config_dict["data_path"], "train")
    config_dict["valid_data_path"] = os.path.join(config_dict["data_path"], "valid")
    if (
        not os.path.exists(config_dict["train_data_path"]) or
        not os.path.exists(config_dict["valid_data_path"])
    ):
        raise ValueError("train and/or valid subdirs are not in data_path!")
    del config_dict["data_path"]

    config_dict["adc_threshold"] = config_dict["adc_threshold"] * config_dict["scalefactors"][0]

    if prep_checkpoint_dir:
        config_dict["checkpoint_dir"] = os.path.join(
            config_dict["checkpoints_dir"], config_dict["name"]
        )
        if not os.path.exists(config_dict["checkpoint_dir"]):
            os.makedirs(config_dict["checkpoint_dir"])
        else:
            print(
                "WARNING: {} already exists, data may be overwritten".format(
                    config_dict["checkpoint_dir"]
                )
            )
        shutil.copyfile(
            config_file, os.path.join(config_dict["checkpoint_dir"], os.path.basename(config_file))
        )
        if not os.path.exists(os.path.join(config_dict["checkpoint_dir"], "preds")):
            os.makedirs(os.path.join(config_dict["checkpoint_dir"], "preds"))


    config_namedtuple = namedtuple("config", config_dict)
    config = config_namedtuple(**config_dict)

    return config
