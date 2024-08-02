
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
from tqdm import tqdm



class EDIMTester:
    """Exclusive Deep Info Max tester

    Args:
        model (EDIM): Exclusive model backbone
        dataset (Dataset): Dataset
        batch_size (int): Batch size
        device (str): Device among cuda/cpu
    """

    def __init__(
        self,
        model: EDIM,
        dataset: Dataset,
        batch_size: int,
        device: str,
    ):

        self.dataloader = DataLoader(dataset, batch_size=batch_size)
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
    
    @staticmethod
    def accuracy(y_pred, target):
        return torch.sum(y_pred == target).item()

    def test(self):
        """Trained excluvise model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """

        with torch.no_grad():

            left_cat_acc = right_cat_acc = elevation_acc = lightning_acc = 0

            for train_batch in tqdm(self.dataloader):
                sample = train_batch
                edim_outputs = self.model.forward_generator(
                    x=sample.left_img.to(self.device), y=sample.right_img.to(self.device)
                )

                edim_classif_outputs = self.model.forward_classifier(
                    edim_outputs=edim_outputs
                )

                left_cat_acc += self.accuracy(
                    y_pred=torch.argmax(edim_classif_outputs.left_cat_logits, dim=1), 
                    target=sample.left_cat.to(self.device)
                )
                right_cat_acc += self.accuracy(
                    y_pred=torch.argmax(edim_classif_outputs.right_cat_logits, dim=1), 
                    target=sample.right_cat.to(self.device)
                )

                elevation_acc += self.accuracy(
                    y_pred=torch.argmax(edim_classif_outputs.elevation_logits, dim=1), 
                    target=sample.elevation.to(self.device)
                )

                lightning_acc += self.accuracy(
                    y_pred=torch.argmax(edim_classif_outputs.lightning_logits, dim=1), 
                    target=sample.lightning.to(self.device)
                )

        print("EDIM ACCURACY")
        print(
            f"left_cat_acc = {left_cat_acc / len(self.dataset):.3f},",
            f"right_cat_acc = {right_cat_acc / len(self.dataset):.3f},",
            f"elevation_acc = {elevation_acc / len(self.dataset):.3f},",
            f"lightning_acc = {lightning_acc / len(self.dataset):.3f}\n",
        )
