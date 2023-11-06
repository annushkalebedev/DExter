import os, sys
import warnings
warnings.filterwarnings("ignore")
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
from renderer import Renderer
import hook

def eval_renderer(cfg, val_loader):
    """run the evaluation set on other renderer for comparison
    - Basis Mixer
    - ScorePerformer
    """

    
    for batch_idx, batch in enumerate(val_loader):

        for idx in range(0, cfg.dataloader.val.batch_size, 2): 

            snote_id_path = batch['snote_id_path'][idx]
            snote_ids = np.load(snote_id_path)
            if len(snote_ids) < 10: # when there is too few notes, the rendering would have problems.
                continue

            piece_name = batch['piece_name'][idx]
            
            mid_out_dir = f"{cfg.task.samples_root}/EVAL-{cfg.renderer}/batch={batch_idx}/"

            if cfg.renderer == 'basismixer':
                mid_out_path = f"{mid_out_dir}/{idx}_{piece_name}.mid"
                os.makedirs(mid_out_dir, exist_ok=True)
                # already modified the original to only render the segment
                os.system(f"python {cfg.renderer_path} {batch['score_path'][idx]} {mid_out_path} {batch['snote_id_path'][idx]}")

                # load the saved label performance from the generation directory
                lpp = f"artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac0.75-L12-C768-cfdg_ddpm-w=1.2-p=0.1-k=3-dia=2-4/epoch=0/batch={batch_idx}/{idx}_{piece_name}_label.mid"
                if os.path.exists(mid_out_path):
                    # generate evaluation file
                    renderer = Renderer(mid_out_dir)
                    renderer.load_external_performances(mid_out_path, batch['score_path'][idx], snote_ids,
                                                        label_performance_path=lpp)
                    renderer.save_performance_features()
                    renderer.save_pf_distribution()
                    hook()
    return 



@hydra.main(config_path="config", config_name="evaluate")
def main(cfg):
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.system("wandb sync --clean-force --clean-old-hours 3")

    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    cfg.data_root = to_absolute_path(cfg.data_root)
    
    # load our data
    paired, _ = load_transfer_pair(K=2000000, N=cfg.seg_len) 
    train_set, valid_set = split_train_valid(paired, select_num=3000)
    assert(len(train_set) % 2 == 0)
    assert(len(valid_set) % 2 == 0)   
    val_loader = DataLoader(valid_set, **cfg.dataloader.val)  


    if cfg.renderer == "diff":
        # Normalize data
        _, valid_set, means, stds = dataset_normalization(train_set, valid_set)
        cfg.task.dataset_means = means
        cfg.task.dataset_stds = stds
        val_loader = DataLoader(valid_set[:cfg.dataloader.num_data], **cfg.dataloader.val)  

        # Model
        model = getattr(Model, cfg.model.name).load_from_checkpoint(
                                            checkpoint_path=cfg.pretrained_path,\
                                            **cfg.model.args, 
                                            **cfg.task)
                
        lw = "".join(str(x) for x in cfg.task.loss_weight)
        name = f"target{cfg.train_target}-lw{lw}-len{cfg.seg_len}-beta{round(cfg.task.beta_end, 2)}-steps{cfg.task.timesteps}-{cfg.task.training.mode}-" + \
                f"Transfer{cfg.task.transfer}-ssfrac{cfg.task.sample_steps_frac}-" + \
                f"L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
                f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
                f"p={cfg.model.args.cond_dropout}-k={cfg.model.args.kernel_size}-" + \
                f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}"

        name = "EVAL-" + name

    
        wandb_logger = WandbLogger(project="DiffPerformer", name=name, save_code=True)   
        trainer = pl.Trainer(**cfg.trainer,
                            logger=wandb_logger,
                            )
        
        trainer.test(model, val_loader)
    else:
        eval_renderer(cfg, val_loader)
    


if __name__ == "__main__":
    main()