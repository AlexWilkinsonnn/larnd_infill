# NOTE This is to make a histogram of fraction of the masked ADC within the candidate coordinates
# given by the reflections
# python reflections_coverage.py  experiments/nu_infillcuts_forwardbackwardzshift038_zdownsample10/exp3_1_reflections.yml
# XXX Should comment out the:
# if torch.all(data["input_feats"][:, -1] == 0):
#     return False
# In set_input or completion_net_adversarial.py when running this

import argparse, os

from tqdm import tqdm
import numpy as np

import torch;
import MinkowskiEngine as ME

from ME.config_parsers.parser_train import get_config
from ME.dataset import LarndDataset, CollateCOO
from ME.models.completion_net_adversarial import CompletionNetAdversarial

FIGSIZE=(6, 5)
FONTSIZE=13
LABELSIZE=10

def main(args):
    conf = get_config(args.config)

    if not os.path.exists(os.path.join(conf.checkpoint_dir, "thesis_plots")):
        os.makedirs(os.path.join(conf.checkpoint_dir, "thesis_plots"))

    model = CompletionNetAdversarial(conf)
    model.eval()

    collate_fn = CollateCOO(
        coord_feat_pairs=(("input_coords", "input_feats"), ("target_coords", "target_feats"))
    )
    dataset_test = LarndDataset(
        conf.test_data_path,
        conf.data_prep_type,
        conf.vmap,
        conf.n_feats_in, conf.n_feats_out,
        conf.scalefactors,
        # ((-1, 2), (-1, 2), (-3, 4)),# conf.xyz_smear_infill,
        # ((-6, 7), (-6, 7), (-6, 7)),# conf.xyz_smear_active,
        # [30, 30, 120], # conf.xyz_max_reflect_distance,
        conf.xyz_smear_infill, conf.xyz_smear_active,
        conf.xyz_max_reflect_distance,
        max_dataset_size=0,
        seed=1
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False
    )

    n_reflections_coords, n_active_coords = [], []
    total_infill_adcs, frac_adc_contained = [], []
    for i_data, data in tqdm(enumerate(dataloader_test), desc="Test Loop"):
        # if i_data > 1000:
        #     break

        if not model.set_input(data):
            continue

        vis = model.get_current_visuals()

        s_target, s_in = vis["s_target"], vis["s_in"]
        s_target = ME.SparseTensor(
            coordinates=s_target.C, features=s_target.F * (1.0 / conf.scalefactors[0]),
            device=torch.device("cpu")
        )
        s_in_F = s_in.F
        s_in_F[:, 0] = s_in_F[:, 0] * (1.0 / conf.scalefactors[0])
        s_in = ME.SparseTensor(
            coordinates=s_in.C, features=s_in_F, device=torch.device("cpu")
        )

        s_in_infill_mask = (s_in.F[:, -1] == 1.0) * (s_in.F[:, 0] == 0.0)
        infill_coords = s_in.C[s_in_infill_mask].type(torch.float)
        s_in_active_mask = (s_in.F[:, 0] != 0.0)
        active_coords = s_in.C[s_in_active_mask].type(torch.float)

        n_reflections_coords.append(len(infill_coords))
        n_active_coords.append(len(active_coords))

        infill_feats_target = s_target.features_at_coordinates(infill_coords)

        contained_infill_adc = infill_feats_target.sum().item()
        total_infill_adc = s_target.F.sum().item() - s_target.features_at_coordinates(active_coords).sum().item()
        frac_adc_contained.append(contained_infill_adc / total_infill_adc if total_infill_adc != 0.0 else 1.0)
        total_infill_adcs.append(total_infill_adc)

    print(np.mean(frac_adc_contained))
    np.save("frac_adc_contained.npy", np.array(frac_adc_contained))
    np.save("total_infill_adcs.npy", np.array(total_infill_adcs))
    np.save("n_reflections_coords.npy", np.array(n_reflections_coords))
    np.save("n_active_coords.npy", np.array(n_active_coords))

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
