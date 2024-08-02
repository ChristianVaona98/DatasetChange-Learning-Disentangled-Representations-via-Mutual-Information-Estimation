import torch.optim as optim
import torch
from losses.EDIM_loss import EDIMLoss
from models.SDIM import SDIM
from losses.SDIM_loss import SDIMLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch as mpy

from utils.custom_typing import SDIMSmallNORBOutputs, SDIMSmallNORBLosses


class SDIMTrainer:
    def __init__(
        self,
        model: SDIM,
        loss: SDIMLoss,
        dataset_train: Dataset,
        learning_rate: float,
        batch_size: int,
        device: str,
    ):
        """Shared Deep Info Max trainer

        Args:
            model (SDIM): Shared model backbone
            loss (SDIMLoss): Shared loss
            dataset_train (Dataset): Train dataset
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            device (str): Device among cuda/cpu
        """
        self.train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
        self.model = model.to(device)
        self.loss = loss
        self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network optimizers
        self.optimizer_encoder_x = optim.Adam(
            model.sh_enc_x.parameters(), lr=learning_rate
        )
        self.optimizer_encoder_y = optim.Adam(
            model.sh_enc_y.parameters(), lr=learning_rate
        )
        self.optimizer_local_stat_x = optim.Adam(
            model.local_stat_x.parameters(), lr=learning_rate
        )
        self.optimizer_local_stat_y = optim.Adam(
            model.local_stat_y.parameters(), lr=learning_rate
        )
        self.optimizer_global_stat_x = optim.Adam(
            model.global_stat_x.parameters(), lr=learning_rate
        )
        self.optimizer_global_stat_y = optim.Adam(
            model.global_stat_y.parameters(), lr=learning_rate
        )


        self.optimizer_cat_classifier = optim.Adam(
            model.cat_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_elevation_classifier = optim.Adam(
            model.elevation_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_lightning_classifier = optim.Adam(
            model.lightning_classifier.parameters(), lr=learning_rate
        )

    def gradient_zero(self):
        """Set all the networks gradient to zero"""
        self.optimizer_encoder_x.zero_grad()
        self.optimizer_encoder_y.zero_grad()

        self.optimizer_local_stat_x.zero_grad()
        self.optimizer_local_stat_y.zero_grad()

        self.optimizer_global_stat_x.zero_grad()
        self.optimizer_global_stat_y.zero_grad()

        self.optimizer_cat_classifier.zero_grad()
        self.optimizer_elevation_classifier.zero_grad()
        self.optimizer_lightning_classifier.zero_grad()

    def compute_gradient(
        self,
        sdim_output: SDIMSmallNORBOutputs,
        cat_labels: torch.Tensor,
        elevation_labels: torch.Tensor,
        lightning_labels: torch.Tensor,
    ) -> SDIMSmallNORBLosses:
        """Compute the SDIM gradient

        Args:
            sdim_output (SDIMSmallNORBOutputs): Shared model outputs
            digit_labels (torch.Tensor): [description]
            color_bg_labels (torch.Tensor): [description]
            color_fg_labels (torch.Tensor): [description]

        Returns:
            SDIMSmallNORBLosses: [Shared model losses value]
        """
        losses = self.loss(
            sdim_outputs=sdim_output,
            cat_labels=cat_labels,
            elevation_labels=elevation_labels,
            lightning_labels=lightning_labels,
        )
        losses.total_loss.backward()
        return losses

    def gradient_step(self):
        """Make an optimisation step for all the networks"""

        self.optimizer_encoder_x.step()
        self.optimizer_encoder_y.step()

        self.optimizer_local_stat_x.step()
        self.optimizer_local_stat_y.step()

        self.optimizer_global_stat_x.step()
        self.optimizer_global_stat_y.step()

        self.optimizer_cat_classifier.step()
        self.optimizer_elevation_classifier.step()
        self.optimizer_lightning_classifier.step()

    def train(self, epochs, xp_name="test"):
        """Trained shared model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """
        mlflow.set_experiment(experiment_name=xp_name)
        with mlflow.start_run() as run:
            mlflow.log_param("Batch size", self.batch_size)
            mlflow.log_param("Learning rate", self.learning_rate)
            mlflow.log_param("Local mutual weight", self.loss.local_mutual_loss_coeff)
            mlflow.log_param("Global mutual weight", self.loss.global_mutual_loss_coeff)
            mlflow.log_param("L1 weight", self.loss.shared_loss_coeff)
            mlflow.log_param("Epochs", epochs)
            log_step = 0
            for epoch in tqdm(range(epochs)):
                for idx, train_batch in enumerate(self.train_dataloader):
                    sample = train_batch
                    self.gradient_zero()
                    sdim_outputs = self.model(
                        x=sample.left_img.to(self.device), y=sample.right_img.to(self.device)
                    )
                    # abbiamo scelto: left cambia elevation e right lightning
                    losses = self.compute_gradient(
                        sdim_output=sdim_outputs,
                        cat_labels=sample.left_cat.to(self.device),
                        elevation_labels=sample.elevation.to(self.device),
                        lightning_labels=sample.lightning.to(self.device),
                    )
                    dict_losses = losses._asdict()
                    mlflow.log_metrics(
                        {k: v.item() for k, v in dict_losses.items()}, step=log_step
                    )
                    log_step += 1
                    self.gradient_step()

            encoder_x_path, encoder_y_path = "sh_encoder_x", "sh_encoder_y"
            try:
                mpy.log_state_dict(self.model.sh_enc_x.state_dict(), encoder_x_path)
                mpy.log_state_dict(self.model.sh_enc_y.state_dict(), encoder_y_path)
            except:
                torch.save(self.model.sh_enc_x.state_dict(), 'sh_encoder_x.pth')
                torch.save(self.model.sh_enc_y.state_dict(), 'sh_encoder_y.pth')
            
            torch.save(self.model.state_dict(), 'SDIM.pth')