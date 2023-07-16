import os, sys, glob
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
import numpy as np
import hook


def render_sample(score, sampled_parameters, snote_ids, save_path):
    """
    render the sample to midi file and save 
        snote_ids (str or ) : 
    """
    if type(snote_ids) == str:
        snote_ids = np.load(snote_ids)
    if type(sampled_parameters) == str:
        sampled_parameters = np.load(sampled_parameters)

    for idx in range(len(sampled_parameters)): # batch
        parameters = sampled_parameters[idx]
        parameters = list(zip(*parameters.T))
        performance_array = np.array(parameters, 
                            dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4")])
        pad_mask = np.full(snote_ids.shape, False)
        performed_part = pt.musicanalysis.decode_performance(score, performance_array, snote_ids=snote_ids, pad_mask=pad_mask)

        pt.save_performance_midi(performed_part, f"{save_path}_{idx}.mid")

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


if __name__ == "__main__":

    score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml")
    performed_part = render_sample(score, 'samples/test_sample.npy', "data/snote_id.npy", "samples/label")

    pass