
import torch
import torch.nn as nn
from models.SDIM import SDIM
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import argparse
import numpy as np
import os
import random
import ruamel.yaml as yaml
from utils.smallnorb_dataloader import SmallNORBDataset
from neural_networks.encoder import BaseEncoder
from tester.sdim_tester import SDIMTester

def run(
    xp_name: str,
    conf_path: str,
    data_base_folder: str,
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
    LOSS_PARAM = conf["loss_param"]

    sdim = SDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=MODEL_PARAM["shared_dim"],
        switched=MODEL_PARAM["switched"],
    )

    sdim.load_state_dict(torch.load("SDIM.pth"))

    dataset = SmallNORBDataset(data_folder=data_base_folder, split='test')

    device = TRAINING_PARAM["device"]
    batch_size = TRAINING_PARAM["batch_size"]
    epochs = TRAINING_PARAM["epochs"]
    tester = SDIMTester(
        dataset=dataset,
        model=sdim,
        batch_size=batch_size,
        device=device,
    )
    tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning Disentangled Representations via Mutual Information Estimation"
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

    args = parser.parse_args()
    xp_name = args.xp_name
    conf_path = args.conf_path
    data_base_folder = args.data_base_folder
    seed = args.seed

    run(
        xp_name=xp_name,
        conf_path=conf_path,
        data_base_folder=data_base_folder,
        seed=seed,
    )