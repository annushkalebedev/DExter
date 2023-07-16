import sys
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
from tqdm import tqdm
import numpy as np
import hydra
from hydra.utils import to_absolute_path
import model as Model
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import hook

import AudioLoader.music.amt as MusicDataset


class SubseqSampler:
    def __init__(self, dataset, seq_len):
        self.dataset = dataset
        self.seq_len = seq_len
    def __getitem__(self, item):
        if self.seq_len == self.dataset.shape[1]:
            return self.dataset[item]
        seq_start = np.random.randint(0, self.dataset.shape[1] - self.seq_len)
        return self.dataset[item][:, seq_start:seq_start+self.seq_len]

    def __len__(self):
        return len(self.dataset)
    

@hydra.main(config_path="config", config_name="unconditioned_render")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    
    # load our data
    midi_data = np.load(cfg.data_root + "/p_codec.npy", allow_pickle=True) # (38140, 1024, 3) (22, 156, 4)
    midi_data = SubseqSampler(midi_data, 328)

    val_idx = int(len(midi_data) * 0.2)
    train_loader = DataLoader(midi_data[val_idx:], **cfg.dataloader.train)
    val_loader = DataLoader(midi_data[:val_idx], **cfg.dataloader.val)    

    # Model
    if cfg.load_trained:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
                                            checkpoint_path=cfg.pretrained_path,\
                                            **cfg.model.args, spec_args=cfg.spec.args, **cfg.task)
    else:
        model = getattr(Model, cfg.model.name)\
            (**cfg.model.args, spec_args=cfg.spec.args, **cfg.task)
            
    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)    
    
    if cfg.model.name == 'DiffRollBaseline':
        name = f"{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"t={cfg.task.time_mode}-x_t={cfg.task.x_t}-k={cfg.model.args.kernel_size}-" + \
               f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}-" + \
               f"{cfg.dataset.name}"
    elif cfg.model.name == 'ClassifierFreeDiffRoll':
        name = f"{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
               f"p={cfg.model.args.spec_dropout}-k={cfg.model.args.kernel_size}-" + \
               f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}-" + \
               f"{cfg.dataset.name}"
    else:
        name = f"{cfg.model.name}-{cfg.task.sampling.type}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"dilation{cfg.model.args.dilation_base}-{cfg.task.loss_type}-{cfg.dataset.name}"
        
    wandb_logger = WandbLogger(project="DiffPerformer", name=name)   

    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=wandb_logger
                         )
    if not cfg.test_only:
        trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)
    
if __name__ == "__main__":
    main()