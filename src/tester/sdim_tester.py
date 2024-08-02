
import torch
import torch.nn as nn
from models.SDIM import SDIM
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import ruamel.yaml as yaml
from tqdm import tqdm



class SDIMTester:
    """Exclusive Deep Info Max tester

    Args:
        model (SDIM): Exclusive model backbone
        dataset (Dataset): Dataset
        batch_size (int): Batch size
        device (str): Device among cuda/cpu
    """

    def __init__(
        self,
        model: SDIM,
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
        # count the right prediction
        return torch.sum(y_pred == target).item()

    def test(self):
        """Trained excluvise model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """
        with torch.no_grad():
            cat_acc = elevation_acc = lightning_acc = 0

            for train_batch in tqdm(self.dataloader):
                sample = train_batch
                sdim_outputs = self.model(
                    x=sample.left_img.to(self.device), y=sample.right_img.to(self.device)
                )

                cat_acc += self.accuracy(
                    y_pred=torch.argmax(sdim_outputs.cat_logits, dim=1), 
                    target=sample.left_cat.to(self.device)
                )

                elevation_acc += self.accuracy(
                    y_pred=torch.argmax(sdim_outputs.elevation_logits, dim=1), 
                    target=sample.elevation.to(self.device)
                )

                lightning_acc += self.accuracy(
                    y_pred=torch.argmax(sdim_outputs.lightning_logits, dim=1), 
                    target=sample.lightning.to(self.device)
                )

        print("SDIM ACCURACY")
        print(
            f"cat_acc = {cat_acc / len(self.dataset):.3f},",
            f"elevation_acc = {elevation_acc / len(self.dataset):.3f},",
            f"lightning_acc = {lightning_acc / len(self.dataset):.3f}\n",
        )
