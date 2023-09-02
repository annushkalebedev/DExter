import os, sys
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import warnings
warnings.filterwarnings('ignore')
import partitura as pt
from tqdm import tqdm
import numpy as np
import hydra
from hydra.utils import to_absolute_path
import model as Model
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import *
import hook
    

@hydra.main(config_path="config", config_name="train")
def main(cfg):
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.system("wandb sync --clean-force --clean-old-hours 3")

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    cfg.data_root = to_absolute_path(cfg.data_root)
    
    # load our data
    codec_data = np.load(cfg.data_root + f"/codec_N={cfg.seg_len}.npy", allow_pickle=True) # (N_data, 1000, 4)
    # N = 1000: 23,190 - stats for ATEPP+ASAP+VIENNA
    # N = 300: 68,981
    # N = 100: 200,315 
    
    # pick the specific group into the first batch of validation 
    train_set, valid_set = split_train_valid(codec_data)
    train_loader = DataLoader(train_set, **cfg.dataloader.train)
    val_loader = DataLoader(valid_set, **cfg.dataloader.val)    

    # Model
    if cfg.load_trained:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
                                            checkpoint_path=cfg.pretrained_path,\
                                            **cfg.model.args, 
                                            **cfg.task)
    else:
        model = getattr(Model, cfg.model.name)(**cfg.model.args, **cfg.task)
            
    
    name = f"len{cfg.seg_len}-beta{round(cfg.task.beta_end, 2)}-steps{cfg.task.timesteps}-{cfg.task.training.mode}-" + \
            f"Transfer{cfg.task.transfer}-ssfrac{cfg.task.sample_steps_frac}-" + \
            f"L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
            f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
            f"p={cfg.model.args.cond_dropout}-k={cfg.model.args.kernel_size}-" + \
            f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}"

    if cfg.test_only:
        name = "TEST-" + name

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint, dirpath=f'artifacts/checkpoint/{name}')    
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