import os, sys, glob
import torch
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
import numpy as np
import hook


def render_sample(sampled_parameters, batch, save_path):
    """
    render the sample to midi file and save 
        sampled_parameters (np.array) : (B, L, 4)
        batch : dictionary
            "score_path" : score_path to load
            "snote_id_path" : snote_id path to load
        snote_ids (str or list) : list of snote_id or path to load
        
    """

    for idx in range(len(sampled_parameters)): # batch iteration
        parameters = sampled_parameters[idx]
        parameters = list(zip(*parameters.T))
        performance_array = np.array(parameters, 
                            dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4")])
        
        snote_ids = np.load(batch['snote_id_path'][idx])
        score = pt.load_musicxml(batch['score_path'][idx])
        piece_name = batch['piece_name']
        N = len(snote_ids)
        
        pad_mask = np.full(snote_ids.shape, False)
        performed_part = pt.musicanalysis.decode_performance(score, performance_array[:N], snote_ids=snote_ids, pad_mask=pad_mask)

        pt.save_performance_midi(performed_part, f"{save_path}_{idx}_{piece_name}.mid")

    return performed_part 


def animate_sampling(t_idx, fig, ax_flat, caxs, noise_list, total_timesteps):
    # noise_list: Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
    # x_t (B, 1, T, F)
    # clearing figures to prevent slow down in each iteration.d
    fig.canvas.draw()
    for idx in range(len(noise_list[0][0])): # visualize only 4 piano rolls in a batch
        ax_flat[idx].cla()
        ax_flat[4+idx].cla()
        caxs[idx].cla()
        caxs[4+idx].cla()     

        # roll_pred (1, T, F)
        im1 = ax_flat[idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
        im2 = ax_flat[4+idx].imshow(noise_list[1 + total_timesteps - t_idx][0][idx][0].T, aspect='auto', origin='lower')
        fig.colorbar(im1, cax=caxs[idx])
        fig.colorbar(im2, cax=caxs[4+idx])

    fig.suptitle(f't={t_idx}')
    row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
    row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')


class Normalization():
    """
    This class is for normalizing the spectrograms batch by batch.
    The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected.
    In this paper, we found that 'imagewise' normalization works better than 'framewise'
    
    If framewise is used, then X must follow the shape of (B, F, T)
    """
    def __init__(self, min, max, mode='imagewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                x_std = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                x_std[torch.isnan(x_std)]=0 # Making nan to 0
                x_scaled = x_std * (max - min) + min
                return x_scaled
        elif mode == 'imagewise':
            def normalize(x):
                # x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                # x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x.flatten(1).max(1, keepdim=True)[0]
                x_min = x.flatten(1).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                x_std = (x-x_min)/(x_max-x_min)
                x_scaled = x_std * (max - min) + min
                x_scaled[torch.isnan(x_scaled)]=min # if piano roll is empty, turn them to min
                return x_scaled
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)



if __name__ == "__main__":

    score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml")
    performed_part = render_sample(score, 'samples/test_sample.npy', "data/snote_id.npy", "samples/label")

    pass