import os, sys, glob
import torch
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
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
    B = len(sampled_parameters) # batch_size

    fig, ax = plt.subplots(int(B/2), 4, figsize=(24, 24))
    for idx in range(B): 
        performance_array = parameters_to_performance_array(sampled_parameters[idx])

        snote_ids = np.load(batch['snote_id_path'][idx])
        score = pt.load_musicxml(batch['score_path'][idx], force_note_ids='keep')
        # unfold the score if necessary (mostly for ASAP)
        if ("-" in snote_ids[0] and 
            "-" not in score.note_array()['id'][0]):
            score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
        piece_name = batch['piece_name'][idx]
        N = len(snote_ids)
        
        pad_mask = np.full(snote_ids.shape, False)
        performed_part = pt.musicanalysis.decode_performance(score, performance_array[:N], snote_ids=snote_ids, pad_mask=pad_mask)

        # compute the tempo curve of sampled parameters (avg for joint-onsets)
        beats, (performed_tempo, label_tempo), (performed_vel, label_vel) = compare_performance_curve(
                                                score, snote_ids, performance_array,
                                                parameters_to_performance_array(batch['p_codec'][idx].cpu()))
        
        ax.flatten()[idx].plot(beats, performed_tempo, label="performed_tempo")
        ax.flatten()[idx].plot(beats, label_tempo, label="label_tempo")
        ax.flatten()[idx].set_ylim(0, 500)

        ax.flatten()[idx+B].plot(beats, performed_vel, label="performed_vel")
        ax.flatten()[idx+B].plot(beats, label_vel, label="label_vel")

        pt.save_performance_midi(performed_part, f"{save_path}/{idx}_{piece_name}.mid")

    return performed_part, fig


def compare_performance_curve(score, snote_ids, pcodec_pred, pcodec_label, visualize=False):
    """compute the performance curve (tempo curve \ velocity curve) from given performance array

    Returns:
        onset_beats : 
        (performed_tempo, label_tempo): 
        (performed_vel, label_vel): 
    """
    na = score.note_array()
    na = na[np.in1d(na['id'], snote_ids)]

    joint_pcodec_pred = rfn.merge_arrays([na, pcodec_pred[:len(na)]], flatten = True, usemask = False)
    onset_beats = np.unique(na['onset_beat'])
    performed_bp = [joint_pcodec_pred[joint_pcodec_pred['onset_beat'] == ob]['beat_period'].mean() for ob in onset_beats]
    performed_vel = [joint_pcodec_pred[joint_pcodec_pred['onset_beat'] == ob]['velocity'].mean() for ob in onset_beats]
    tempo_curve_pred = interp1d(onset_beats, 60 / np.array(performed_bp))

    joint_pcodec_label = rfn.merge_arrays([na, pcodec_label[:len(na)]], flatten = True, usemask = False)
    onset_beats = np.unique(na['onset_beat'])
    label_bp = [joint_pcodec_label[joint_pcodec_label['onset_beat'] == ob]['beat_period'].mean() for ob in onset_beats]
    label_vel = [joint_pcodec_label[joint_pcodec_label['onset_beat'] == ob]['velocity'].mean() for ob in onset_beats]
    tempo_curve_label = interp1d(onset_beats, 60 / np.array(label_bp))

    return onset_beats, (60 / np.array(performed_bp), 60 / np.array(label_bp)), (np.array(performed_vel), np.array(label_vel))



def parameters_to_performance_array(parameters):
    """
        parameters (np.ndarray) : shape (1000, 5)
    """
    parameters = list(zip(*parameters.T))
    performance_array = np.array(parameters, 
                        dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4"), ("pedal", "f4")])

    return performance_array


TESTING_GROUP = [
    '07403_seg0',  # bethoven appasionata beginning
    '10696_seg0',  # Jeux D'eau beginning
    '11579_seg4',  # Chopin Ballade 2 transitioning passage
    '00129_seg0',  # Rachmaninov etude tableaux no.5 eb minor
    'ASAP_Schumann_Arabeske_seg0',         # Schumann Arabeske beginning
    'ASAP_Bach_Fugue_bwv_867_seg0',         # Bach fugue 867 beginning 
    'ASAP_Mozart_Piano_Sonatas_8-1_seg0',  # Mozart a minor beginning
    'VIENNA422_Schubert_D783_no15_seg0',   # Vienna422 schubert piece beginning
]

def split_train_valid(codec_data):
    """select the specific group into the first batch of validation data 
    Returns the train set and valid set as list, the first 8 in the valid set is the selected samples
    """

    selected_cd, unselected_cd = defaultdict(list), []
    for cd in codec_data:
        for name in TESTING_GROUP:
            if (not selected_cd[name]) and name in cd['snote_id_path']:
                selected_cd[name] = cd
                break
        else:
            unselected_cd.append(cd)
            
    train_idx = int(len(codec_data) * 0.85)
    np.random.shuffle(unselected_cd)
    train_set = unselected_cd[:train_idx]
    valid_set = list(selected_cd.values()) + unselected_cd[train_idx:]

    return train_set, valid_set

def plot_codec(codec1, codec2, ax0, ax1, fig):
    # plot the pred p_codec and label p_codec, on the given two axes

    p_im = ax0.imshow(codec1.T, aspect='auto', origin='lower')
    ax0.set_yticks([0, 1, 2, 3, 4])
    ax0.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log", "pedal"])
    fig.colorbar(p_im, orientation='vertical', ax=ax0)

    s_im = ax1.imshow(codec2.T, aspect='auto', origin='lower')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log"])
    fig.colorbar(s_im, orientation='vertical', ax=ax1)
    
    return 


def animate_sampling(t_idx, fig, ax_flat, caxs, noise_list, total_timesteps):
    # noise_list: Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
    # x_t (B, 1, T, F)
    # clearing figures to prevent slow down in each iteration.d

    fig.canvas.draw()
    for idx in range(len(noise_list[0][0])): # visualize 8 samples in the batch
        ax_flat[2*idx].cla()
        ax_flat[2*idx+1].cla()
        caxs[2*idx].cla()
        caxs[2*idx+1].cla()     

        # roll_pred (1, T, F)
        im1 = ax_flat[2*idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
        im2 = ax_flat[2*idx+1].imshow(noise_list[1 + total_timesteps - t_idx][0][idx][0].T, aspect='auto', origin='lower')
        fig.colorbar(im1, cax=caxs[2*idx])
        fig.colorbar(im2, cax=caxs[2*idx+1])

    fig.suptitle(f't={t_idx}')
    row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
    row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')


class Normalization():
    """
    This class is for normalizing the input batch by batch.
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

    performed_part = render_sample(score, 'samples/test_sample.npy', "data/snote_id.npy", "samples/label")

    pass