
import torch
import torch.nn as nn
from models.EDIM import EDIM
from utils.custom_typing import (
    EDIMClassifLosses,
    EDIMClassifierOutputs,
    DiscrLosses,
    GenLosses,
    GeneratorOutputs,
    DiscriminatorOutputs,
    EDIMSmallNORBOutputs,
)
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import argparse
import numpy as np
import os
import random
import ruamel.yaml as yaml
from models.EDIM import EDIM
from utils.smallnorb_dataloader import SmallNORBDataset
from neural_networks.encoder import BaseEncoder
from tester.edim_tester import EDIMTester

def run(
    xp_name: str,
    conf_path: str,
    data_base_folder: str,
    trained_enc_x_path: str,
    trained_enc_y_path: str,
    seed: int = None,
):
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)
    if seed is not None:
        seed = seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    TRAINING_PARAM = conf["training_param"]
    MODEL_PARAM = conf["model_param"]
    SHARED_PARAM = conf["shared_param"]

    trained_enc_x = BaseEncoder(
        img_feature_size=SHARED_PARAM["feature_size"],
        in_channels=MODEL_PARAM["channels"],
        num_filters=64,
        kernel_size=4,
        repr_dim=SHARED_PARAM["shared_dim"],
    )
    
    trained_enc_y = BaseEncoder(
        img_feature_size=SHARED_PARAM["feature_size"],
        in_channels=MODEL_PARAM["channels"],
        num_filters=64,
        kernel_size=4,
        repr_dim=SHARED_PARAM["shared_dim"],
    )
    trained_enc_x.load_state_dict(torch.load(trained_enc_x_path))
    trained_enc_y.load_state_dict(torch.load(trained_enc_y_path))

    edim = EDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=SHARED_PARAM["shared_dim"],
        exclusive_dim=MODEL_PARAM["exclusive_dim"],
        trained_encoder_x=trained_enc_x,
        trained_encoder_y=trained_enc_y,
    )

    edim.load_state_dict(torch.load("EDIM.pth"))

    dataset = SmallNORBDataset(data_folder=data_base_folder, split='test')

    device = TRAINING_PARAM["device"]
    batch_size = TRAINING_PARAM["batch_size"]
    tester = EDIMTester(
        dataset=dataset,
        model=edim,
        batch_size=batch_size,
        device=device,
    )
    tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing Disentangled Representations via Mutual Information Estimation"
    )
    parser.add_argument(
        "--xp_name",
        nargs="?",
        type=str,
        default="Shared_training",
        help="Mlflow experiment name",
    )
    parser.add_argument(
        "--conf_path", nargs="?", type=str, default=None, help="Configuration file"
    )
    parser.add_argument(
        "--data_base_folder", nargs="?", type=str, default=None, help="Data folder"
    )

    parser.add_argument("--seed", nargs="?", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--trained_enc_x_path",
        nargs="?",
        type=str,
        default=None,
        help="Pretrained shared encoder x",
    )
    parser.add_argument(
        "--trained_enc_y_path",
        nargs="?",
        type=str,
        default=None,
        help="Pretrained shared encoder y",
    )

    args = parser.parse_args()
    xp_name = args.xp_name
    conf_path = args.conf_path
    data_base_folder = args.data_base_folder
    seed = args.seed
    trained_enc_x_path = args.trained_enc_x_path
    trained_enc_y_path = args.trained_enc_y_path
    run(
        xp_name=xp_name,
        conf_path=conf_path,
        data_base_folder=data_base_folder,
        trained_enc_x_path=trained_enc_x_path,
        trained_enc_y_path=trained_enc_y_path,
        seed=seed,
    )