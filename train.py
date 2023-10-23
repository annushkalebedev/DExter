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
    if cfg.train_target == "transfer": # load only from paired set. 
        paired, _ = load_transfer_pair(K=2000000, N=cfg.seg_len) 
        train_set, valid_set = split_train_valid(paired, select=False)
        assert(len(train_set) % 2 == 0)
        assert(len(valid_set) % 2 == 0)   
        cfg.dataloader.train.shuffle = False
        train_set, valid_set = train_set[:120000], valid_set[:15000]
    else: # generation: keep 2000 pairs in validation but use unpaired for training.
        paired, unpaired = load_transfer_pair(K=2000, N=cfg.seg_len) 
        assert(len(paired) % 2 == 0)  # ~120 batch
        train_set, valid_set = unpaired, paired

    # Normalize data
    train_set, valid_set, means, stds = dataset_normalization(train_set, valid_set)
    cfg.task.dataset_means = means
    cfg.task.dataset_stds = stds

    train_loader = DataLoader(train_set, **cfg.dataloader.train)
    val_loader = DataLoader(valid_set, **cfg.dataloader.val) 

    # set the fraction so that around 20 batchs are output for listen.
    cfg.task.valid_fraction = 20 / len(val_loader)      

    # Model
    if cfg.load_trained:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
                                            checkpoint_path=cfg.pretrained_path,\
                                            **cfg.model.args, 
                                            **cfg.task)
    else:
        model = getattr(Model, cfg.model.name)(**cfg.model.args, **cfg.task)
            
    lw = "".join(str(x) for x in cfg.task.loss_weight)
    name = f"target{cfg.train_target}-lw{lw}-len{cfg.seg_len}-beta{round(cfg.task.beta_end, 2)}-steps{cfg.task.timesteps}-{cfg.task.training.mode}-" + \
            f"Transfer{cfg.task.transfer}-ssfrac{cfg.task.sample_steps_frac}-" + \
            f"L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
            f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
            f"p={cfg.model.args.cond_dropout}-k={cfg.model.args.kernel_size}-" + \
            f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}"

    if cfg.test_only:
        name = "TEST-" + name

    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint, dirpath=f'artifacts/checkpoint/{name}')    
    wandb_logger = WandbLogger(project="DiffPerformer", name=name, save_code=True)   
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=wandb_logger,
                        #  stochastic_weight_avg=True
                         )
    if not cfg.test_only:
        trainer.fit(model, train_loader, val_loader)
    
    trainer.validate(model, val_loader)
    
if __name__ == "__main__":
    main()