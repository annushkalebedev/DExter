import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import MIDICepDataloader


# Define the CNN architecture with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class MIDItoCEP(nn.Module):
    def __init__(self):
        super(MIDItoCEP, self).__init__()
        self.layer1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.fc = nn.Linear(800 * 131 * 64, 7)  # Adjust the input features to match output of flattened resblock

    def forward(self, x):
        x = self.layer1(x.float())
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Define the LightningModule for training and validation
class MIDItoCEPModel(pl.LightningModule):
    def __init__(self):
        super(MIDItoCEPModel, self).__init__()
        self.model = MIDItoCEP().float()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        matrix, cep = batch['matrix'].float(), batch['cep'].float()
        predicted_cep = self(matrix)
        loss = F.mse_loss(predicted_cep, cep)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        matrix, cep = batch['matrix'], batch['cep']
        predicted_cep = self(matrix)
        loss = F.mse_loss(predicted_cep, cep)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        dataset = MIDICepDataloader(mode='train')
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def val_dataloader(self):
        dataset = MIDICepDataloader(mode='test')
        return DataLoader(dataset, batch_size=16)


if __name__ == "__main__":
    # Set up Weights & Biases logging and checkpointing
    wandb_logger = WandbLogger(project="MIDI-CEP-Regression")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # Initialize the model
    model = MIDItoCEPModel()

    # Initialize the Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=100,
        gpus=[3]
    )

    # Train the model
    trainer.fit(model)

    # After training, you can load the best model
    best_model_path = checkpoint_callback.best_model_path
    best_model = MIDItoCEPModel.load_from_checkpoint(checkpoint_path=best_model_path)
