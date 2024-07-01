# NOTE
# Started doing this but I think it is a bad idea due to input data being integer. The transform
# then inverse transform does not produce the original distribution. Just going to not do this.

import argparse, os
from collections import defaultdict

import sparse
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from sklearn.preprocessing import QuantileTransformer

def main(args):
    train_feats = []
    for f in tqdm(os.listdir(args.train_dir)):
        coo = sparse.load_npz(os.path.join(args.train_dir, f))
        coords_feats = defaultdict(lambda : [None] * coo.shape[-1])
        for coord_0, coord_1, coord_2, coord_feat, feat in zip(
            coo.coords[0], coo.coords[1], coo.coords[2], coo.coords[3], coo.data
        ):
            coords_feats[(coord_0, coord_1, coord_2)][coord_feat] = feat
        train_feats.append(np.array(list(coords_feats.values())))

    train_feats = np.concatenate(train_feats)

    qt = QuantileTransformer(output_distribution="uniform")
    qt.fit(train_feats)

    plt.hist(train_feats[:, 0], bins=100)
    plt.show()
    train_feats = qt.transform(train_feats)
    plt.hist(train_feats[:, 0], bins=100)
    plt.show()
    train_feats = qt.inverse_transform(train_feats)
    plt.hist(train_feats[:, 0], bins=100)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir", type=str)
    parser.add_argument("val_dir", type=str)
    parser.add_argument("output_dir", type=str)

    parser.add_argument("--plot", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quantile", action="store_true")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    main(args)

