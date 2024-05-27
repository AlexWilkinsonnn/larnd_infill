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
    "model_params" : {},
    "optimizer_G_params" : {},
    "optimizer_D_params" : {},
    "D_training_stopper" : {},
    "D_type" : "vanilla",
    "D_pause_until_epoch": 0,
    "optimizer_D" : "SGD",
    "optimizer_G" : "SGD",
    "train_script" : "train_sigmask",
    "fake_label" : 0.0,
    "real_label" : 1.0,
    "net_D" : "PatchGAN",
    "save_model" : "never",
    "load_G" : "",
    "load_D" : "",
    "refresh_masks_epoch" : 1,
    "conf_refining_from" : "",
    "weights_refining_from" : ""
}

mandatory_fields = {
    "vmap_path", "data_path",
    "data_prep_type",
    "scalefactors",
    "n_feats_in", "n_feats_out",
    "max_dataset_size",
    "max_valid_dataset_size",
    "batch_size",
    # "initial_lr",
    "loss_func",
    "epochs",
    "lr_decay_iter",
    "checkpoints_dir",
    "name"
}

def get_config(conf_file, overwrite_dict={}, prep_checkpoint_dir=True):
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

    if conf_dict["refresh_masks_epoch"] > 1:
        if conf_dict["data_prep_type"] == DataPrepType.REFLECTION_NORANDOM:
            print(
                "data_prep_type 'reflection_norandom' already caches. " +
                "Setting refresh_masks_epoch to 1"
            )
            conf_dict["refresh_masks_epoch"] = 1
        elif conf_dict["max_num_workers"] != 0:
            raise ValueError(
                "Cache will not work with multiprocessing, " +
                "must have max_num_workers 0 if refresh_masks_epoch not 1"
            )

    conf_dict["train_data_path"] = os.path.join(conf_dict["data_path"], "train")
    conf_dict["valid_data_path"] = os.path.join(conf_dict["data_path"], "valid")
    if (
        not os.path.exists(conf_dict["train_data_path"]) or
        not os.path.exists(conf_dict["valid_data_path"])
    ):
        raise ValueError("train and/or valid subdirs are not in data_path!")
    del conf_dict["data_path"]

    if conf_dict["adc_threshold"] is not None:
        conf_dict["adc_threshold"] = conf_dict["adc_threshold"] * conf_dict["scalefactors"][0]

    if conf_dict["save_model"] not in ["never", "latest", "best", "all"]:
        raise ValueError(
            "'save_model': {} invalid, choose 'never', 'latest', 'best', 'all'".format(
                conf_dict["save_model"]
            )
        )

    if prep_checkpoint_dir:
        conf_dict["checkpoint_dir"] = os.path.join(conf_dict["checkpoints_dir"], conf_dict["name"])
        if not os.path.exists(conf_dict["checkpoint_dir"]):
            os.makedirs(conf_dict["checkpoint_dir"])
        else:
            print(
                "WARNING: {} already exists, data may be overwritten".format(
                    conf_dict["checkpoint_dir"]
                )
            )
        shutil.copyfile(
            conf_file, os.path.join(conf_dict["checkpoint_dir"], os.path.basename(conf_file))
        )
        if not os.path.exists(os.path.join(conf_dict["checkpoint_dir"], "preds")):
            os.makedirs(os.path.join(conf_dict["checkpoint_dir"], "preds"))

    if conf_dict["train_script"] == "train_refine":
        if not conf_dict["conf_refining_from"]:
            raise ValueError("Specify 'conf_refining_from' when doing 'train_refine'")
        if not conf_dict["weights_refining_from"]:
            raise ValueError("Specify 'weights_refining_from' when doing 'train_refine'")

    conf_namedtuple = namedtuple("conf", conf_dict)
    conf = conf_namedtuple(**conf_dict)

    return conf

