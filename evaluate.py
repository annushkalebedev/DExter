import os, sys
from collections import defaultdict
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
    - Basis Mixer: render the passage 
    - ScorePerformer: save generated performances from their colab and match (since they use ASAP as well)
    """
    
    sip_dict = defaultdict(bool) # keeping track of pieces so don't do repetitive computation.

    for batch_idx, batch in tqdm(enumerate(val_loader)):

        if batch_idx == 0:
            continue

        # iterrate our batch. (since in pairs we only print one for the external renderers)
        for idx in range(0, cfg.dataloader.val.batch_size, 2): 

            if idx != 2:
                continue

            snote_id_path = batch['snote_id_path'][idx]

            if sip_dict[snote_id_path]: 
                continue
            sip_dict[snote_id_path] = False

            snote_ids = np.load(snote_id_path)
            if len(snote_ids) < 10: # when there is too few notes, the rendering would have problems.
                continue
            piece_name = batch['piece_name'][idx]

            mid_out_dir = f"{cfg.task.samples_root}/EVAL-{cfg.renderer}/batch={batch_idx}/"
            os.makedirs(mid_out_dir, exist_ok=True)
            
            # load multiple label performances to compare
            snote_id_dir = snote_id_path.split("/")[-1][:-4]
            lpps = glob.glob(f"artifacts/samples/GT/{snote_id_dir}/*.mid")

            save_seg, merge_tracks = False, False
            if cfg.renderer == 'basismixer':
                pred_mid_path = f"{mid_out_dir}/{idx}_{piece_name}.mid"
                # already modified the original to only render the segment
                os.system(f"python {cfg.renderer_path} {batch['score_path'][idx]} {pred_mid_path} {batch['snote_id_path'][idx]}")
            
            if cfg.renderer == "scoreperformer":
                # scoreperformer output are pre-computed from their colab 
                save_seg, merge_tracks = True, True
                pred_mid_path = f"artifacts/samples/EVAL-scoreperformer/{piece_name}.midi" 
                # if os.path.exists(f"{mid_out_dir}/{idx}_{piece_name}.mid"): # don't compute the existing ones.
                #     continue     

            if cfg.renderer == "virtuosonet":
                # virtuosonet output are pre-computed from running their repository
                save_seg = True
                pred_mid_path = f"artifacts/samples/EVAL-virtuosonet/{piece_name}.mid" 
                # if os.path.exists(f"{mid_out_dir}/{idx}_{piece_name}.mid"): # don't compute the existing ones.
                #     continue  

            if cfg.renderer == "dexter":
                # dexter was first rendered in testing step. But this step compare it with all GTs.
                dexter_midiout_path = f"artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferFalse-ssfrac1-L12-C768-cfdg_ddpm-w=1.2-p=0.1-k=3-dia=2-4/epoch=0/batch={batch_idx}/"
                pred_mid_path = f"{dexter_midiout_path}/{idx}_{piece_name}.mid"
                # if os.path.exists(f"{mid_out_dir}/{idx}_{piece_name}_feats_pred.csv"): # don't compute the existing ones.
                #     continue    

            if cfg.renderer == "dexter1":
                # dexter was first rendered in testing step. But this step compare it with all GTs.
                dexter_midiout_path = f"artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac1-L12-C768-cfdg_ddpm-w=1.2-p=0.1-k=3-dia=2-4/epoch=0/batch={batch_idx}/"
                pred_mid_path = f"{dexter_midiout_path}/{idx}_{piece_name}.mid"

            if cfg.renderer == "dexter34":
                # dexter was first rendered in testing step. But this step compare it with all GTs.
                dexter_midiout_path = f"artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac0.75-L12-C768-cfdg_ddpm-w=1.2-p=0.1-k=3-dia=2-4/epoch=0/batch={batch_idx}/"
                pred_mid_path = f"{dexter_midiout_path}/{idx}_{piece_name}.mid"

            if cfg.renderer == "dexter12":
                # dexter was first rendered in testing step. But this step compare it with all GTs.
                dexter_midiout_path = f"artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac0.5-L12-C768-cfdg_ddpm-w=1.2-p=0.1-k=3-dia=2-4/epoch=0/batch={batch_idx}/"
                pred_mid_path = f"{dexter_midiout_path}/{idx}_{piece_name}.mid"

 
            if cfg.renderer == "dexter14":
                # dexter was first rendered in testing step. But this step compare it with all GTs.
                dexter_midiout_path = f"artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac0.25-L12-C768-cfdg_ddpm-w=1.2-p=0.1-k=3-dia=2-4/epoch=0/batch={batch_idx}/"
                pred_mid_path = f"{dexter_midiout_path}/{idx}_{piece_name}.mid"


            for lpp in lpps:
                if os.path.exists(pred_mid_path) and os.path.exists(lpp):
                    # try:
                        # generate evaluation file
                        renderer = Renderer(mid_out_dir, idx=idx)
                        renderer.load_external_performances(pred_mid_path, batch['score_path'][idx], snote_ids,
                                                            label_performance_path=lpp, piece_name=piece_name, save_seg=save_seg, merge_tracks=merge_tracks)
                        
                        renderer.save_performance_features()
                        renderer.save_pf_distribution()
                    # except Exception as e:
                    #     print(e)
            # compute the distribution in regards to the overall GT space.
            try:
                renderer.save_pf_distribution(gt_space=f"artifacts/samples/GT/{snote_id_dir}")
            except Exception as e:
                print(e)            
        


def save_all_gt(cfg, valid_set, indices_dict):
    """save all the segments of validation set ground truth. Grouped by piece seg. """

    for sip, indices in tqdm(indices_dict.items()):
        print(sip, indices)
        if os.path.exists(f"{cfg.task.samples_root}/GT/{sip}"):
            continue
        if not os.path.exists(f"{cfg.task.samples_root}/GT/{sip}"):
            os.makedirs(f"{cfg.task.samples_root}/GT/{sip}", exist_ok=True)
            for idx in indices:
                data = valid_set[idx]
                try:
                    renderer = Renderer(f"{cfg.task.samples_root}/GT/{sip}", 
                                        data['p_codec'],  
                                        label_data=data,
                                        idx=idx)
                    renderer.render_sample()
                    renderer.save_performance_features()
                except Exception as e:
                    print(e)
                    continue
        # get the mean and std of all versions of human performances. as well as concatenation
        for typename in ['feats_pred', 'tv_feats']:
            try:
                csvs = glob.glob(f"{cfg.task.samples_root}/GT/{sip}/*_{typename}.csv")
                tables = pd.concat([pd.read_csv(c) for c in csvs])
                tables_group = tables.groupby(level=0)
                tables_group.mean().to_csv(f"{cfg.task.samples_root}/GT/{sip}/{typename}_mean.csv", index=False)
                tables_group.std().to_csv(f"{cfg.task.samples_root}/GT/{sip}/{typename}_std.csv", index=False)
                tables.to_csv(f"{cfg.task.samples_root}/GT/{sip}/{typename}_all.csv", index=False)
            except:
                print(sip)


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
    train_set, valid_set = split_train_valid(paired, 
                                             select_num=3000,
                                             paired_input=True
                                             )
    # indices_dict = group_same_seg(valid_set)
    # save_all_gt(cfg, valid_set, indices_dict)
    # hook()
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

        if cfg.condition_eval:
            name = "EVALo-" + name
        else:
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