import os, sys, glob
import torch
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
import torch.nn.functional as F
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from prepare_data import *
import hook


def render_sample(sampled_parameters, batch, save_path, 
                  with_source=False, save_interpolation=False,
                  means=None, stds=None):
    """
    render the sample to midi file and save 
        sampled_parameters (np.array) : (B, L, 4)
        batch : dictionary
            "score_path" : score_path to load
            "snote_id_path" : snote_id path to load
        save_path: root directory to save
        with_source: whether source is provided 
        save_interpolation: weather 
        
    """
    B = len(sampled_parameters) 

    # rescale the predictions and labels back to normal
    sampled_parameters = p_codec_scale(sampled_parameters, means, stds)
    batch['p_codec'] = p_codec_scale(batch['p_codec'], means, stds)

    fig, ax = plt.subplots(int(B/2), 4, figsize=(24, 3*B))
    for idx in range(B): 
        performance_array = parameters_to_performance_array(sampled_parameters[idx])

        # update the snote_id_path to the new one
        # snote_id_path = batch['snote_id_path'][idx].replace("data", "/import/c4dm-datasets-ext/DiffPerformer")
        snote_id_path = batch['snote_id_path'][idx]
        snote_ids = np.load(snote_id_path)
        score = pt.load_musicxml(batch['score_path'][idx], force_note_ids='keep')
        # unfold the score if necessary (mostly for ASAP)
        if ("-" in snote_ids[0] and 
            "-" not in score.note_array()['id'][0]):
            score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
        piece_name = batch['piece_name'][idx]
        N = len(snote_ids)
        
        pad_mask = np.full(snote_ids.shape, False)
        performed_part = pt.musicanalysis.decode_performance(score, performance_array[:N], snote_ids=snote_ids, pad_mask=pad_mask)

        pcodec_label = parameters_to_performance_array(batch['p_codec'][2*idx].cpu())
        pcodec_source = parameters_to_performance_array(batch['p_codec'][2*idx+1].cpu())

        if save_interpolation:
            pcodec_interpolate = torch.lerp(batch['p_codec'][idx].cpu(), batch['p_codec'][idx+B].cpu(), 0.5)
            pcodec_interpolate = parameters_to_performance_array(pcodec_interpolate)
            performed_part_interpolate = pt.musicanalysis.decode_performance(score, pcodec_interpolate[:N], snote_ids=snote_ids, pad_mask=pad_mask)
            performed_part_label = pt.musicanalysis.decode_performance(score, pcodec_label[:N], snote_ids=snote_ids, pad_mask=pad_mask)
            performed_part_source = pt.musicanalysis.decode_performance(score, pcodec_source[:N], snote_ids=snote_ids, pad_mask=pad_mask)
            try:
                pt.save_performance_midi(performed_part_interpolate, f"{save_path}/{idx}_{piece_name}_lerp.mid")
                pt.save_performance_midi(performed_part_label, f"{save_path}/{idx}_{piece_name}_label.mid")
                pt.save_performance_midi(performed_part_source, f"{save_path}/{idx}_{piece_name}_source.mid")
                print("done")
            except Exception as e:
                print(e)

        if with_source:
            # compute the tempo curve of sampled parameters (avg for joint-onsets)
            beats, performed_tempo, performed_vel, label_tempo, label_vel, source_tempo, source_vel = compare_performance_curve(
                                                    score, snote_ids, performance_array,
                                                    pcodec_label, 
                                                    pcodec_source=pcodec_source)
        else:
            beats, performed_tempo, performed_vel, label_tempo, label_vel = compare_performance_curve(
                                                    score, snote_ids, performance_array,
                                                    pcodec_label)


        tempo_vel_loss = F.l1_loss(torch.tensor(performed_tempo), torch.tensor(label_tempo)) + \
                                F.l1_loss(torch.tensor(performed_vel), torch.tensor(label_vel)) 

        tempo_vel_cor = pearsonr(performed_tempo, label_tempo)[0] + pearsonr(performed_vel, label_vel)[0]

        ax.flatten()[idx].plot(beats, performed_tempo, label="performed_tempo")
        ax.flatten()[idx].plot(beats, label_tempo, label="label_tempo")
        ax.flatten()[idx].set_ylim(0, 500)

        ax.flatten()[idx+B].plot(beats, performed_vel, label="performed_vel")
        ax.flatten()[idx+B].plot(beats, label_vel, label="label_vel")

        if with_source:
            ax.flatten()[idx].plot(beats, source_tempo, label="source_tempo")
            ax.flatten()[idx+B].plot(beats, source_vel, label="source_vel")

        ax.flatten()[idx].legend()
        ax.flatten()[idx].set_title(f"tempo: {piece_name}")   
        ax.flatten()[idx+B].legend()
        ax.flatten()[idx+B].set_title(f"vel: {piece_name}")

        try:
            pt.save_performance_midi(performed_part, f"{save_path}/{idx}_{piece_name}.mid")
        except Exception as e:
            print(e)

    return performed_part, fig, tempo_vel_loss, tempo_vel_cor


def compare_performance_curve(score, snote_ids, pcodec_pred, pcodec_label, pcodec_source=None):
    """compute the performance curve (tempo curve \ velocity curve) from given performance array
        pcodec_original: another parameter curve, coming from the optional starting point of transfer

    Returns:
        onset_beats : 
        performed_tempo
        label_tempo
        source_tempo
        performed_vel
        label_vel
        source_vel 
        (performed_vel, label_vel): 
    """
    na = score.note_array()
    na = na[np.in1d(na['id'], snote_ids)]

    pcodecs = [pcodec_pred, pcodec_label]
    if type(pcodec_source) != type(None):
        pcodecs.append(pcodec_source)

    onset_beats = np.unique(na['onset_beat'])
    res = [onset_beats]
    for pcodec in pcodecs:

        joint_pcodec = rfn.merge_arrays([na, pcodec[:len(na)]], flatten = True, usemask = False)
        bp = [joint_pcodec[joint_pcodec['onset_beat'] == ob]['beat_period'].mean() for ob in onset_beats]
        vel = [joint_pcodec[joint_pcodec['onset_beat'] == ob]['velocity'].mean() for ob in onset_beats]
        tempo_curve_pred = interp1d(onset_beats, 60 / np.array(bp))
        res.extend([60 / np.array(bp), np.array(vel)])

    return res



def parameters_to_performance_array(parameters):
    """
        parameters (np.ndarray) : shape (1000, 5)
    """
    parameters = list(zip(*parameters.T))
    performance_array = np.array(parameters, 
                        dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4"), ("pedal", "f4")])

    return performance_array


TESTING_GROUP = [
    '07935_seg0',  # bethoven appasionata beginning
    '10713_seg0',  # Jeux D'eau beginning
    '11579_seg3',  # Chopin Ballade 2 transitioning passage
    '00129_seg0',  # Rachmaninov etude tableaux no.5 eb minor
    'ASAP_Schumann_Arabeske_seg0',         # Schumann Arabeske beginning
    'ASAP_Bach_Fugue_bwv_867_seg0',        # Bach fugue 867 beginning 
    'ASAP_Mozart_Piano_Sonatas_12-1_seg0',  # Mozart a minor beginning
    'VIENNA422_Schubert_D783_no15_seg0',   # Vienna422 schubert piece beginning
    #### second group of testing data: paired another performance of the test
    '07925_seg0',  
    '10717_seg0', 
    '11587_seg3',  
    '00130_seg0',  
    'ASAP_Schumann_Arabeske_seg0.',         # select another data example with the same snote_id_path
    'ASAP_Bach_Fugue_bwv_867_seg0.',        
    'ASAP_Mozart_Piano_Sonatas_12-1_seg0.',  
    'VIENNA422_Schubert_D783_no15_seg0.',    
]

def split_train_valid(codec_data, select=True):
    """select the specific group into the first batch of validation data 
    Returns the train set and valid set as list, the first group in the valid set is the selected samples (8 + 8 pair)
    """

    train_idx = int(len(codec_data) * 0.85)
    if not select:
        return codec_data[:train_idx], valid_set[train_idx:]

    selected_cd, unselected_cd = defaultdict(list), []
    for cd in codec_data:
        for name in TESTING_GROUP:
            if (not selected_cd[name]) and name in cd['snote_id_path']:
                selected_cd[name] = cd
                break
        else:
            unselected_cd.append(cd)
            
    
    np.random.shuffle(unselected_cd)
    train_set = unselected_cd[:train_idx]
    valid_set = list(selected_cd.values()) + unselected_cd[train_idx:]

    return train_set, valid_set

def make_transfer_pair(codec_data, K=50000, N=200):
    """make transfer pair from the codec data for the testing set. 
        Transfer data come from real performance (not mix-up combinations), and the pieces used in testing set doesn't go into training. 

        returns:
        - transfer_pairs: list of tuple of codec
        - unpaired: list of single codec

        stats:
        - in total 21272 pairs can be found. In testing we can use ~1000 pairs (K). rest can go into unpaired for training. For full transfer 
            training, we use full pairs. ()
    """
    if os.path.exists(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy"):
        transfer_pairs = np.load(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy", allow_pickle=True)
        unpaired = np.load(f"{BASE_DIR}/codec_N={N}_mixup_unpaired_K={K}.npy", allow_pickle=True)
        return transfer_pairs, unpaired

    transfer_pairs = []

    codec_data = np.array(codec_data)
    codec_data_ = codec_data[list(map(lambda x: "mu" not in x['piece_name'], codec_data))] # only consider those not in mixup
    np.random.shuffle(codec_data_)
    unpaired = codec_data

    for cd in tqdm(codec_data_):
        # if len(transfer_pairs) > K:
        #     break
        seg_id = cd['snote_id_path'].split("seg")[-1].split(".")[0]
        # find the one that belongs to the same piece, same segment number but not itself
        mask = list(map(lambda x: ((
                                    x['score_path'] == cd['score_path']) 
                                   and (f"seg{seg_id}." in x['snote_id_path']) 
                                   and (x['p_codec'] != cd['p_codec']).all() 
                                   ), codec_data_))
        same_piece_cd = codec_data_[mask]
        if len(same_piece_cd):
            for scd in same_piece_cd:
                transfer_pairs.extend([cd, scd])
            # remove all this piece's segments from unpaired list (mixup as well, to prevent leakage)
            unpaired = unpaired[list(map(lambda x: x['score_path'] != cd['score_path'], unpaired))]


    # shuffle by each pair and recover
    transfer_pairs = np.array(transfer_pairs).reshape(2, -1, order='F')
    np.random.shuffle(transfer_pairs.T) 
    transfer_pairs = transfer_pairs.ravel(order='F')
    
    hook()
    np.save(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy", transfer_pairs)
    np.save(f"{BASE_DIR}/codec_N={N}_mixup_unpaired_K={K}.npy", unpaired)

    
    return transfer_pairs, unpaired


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


def compile_condition(s_codec, c_codec):
    """compile s_codec and c_codec into a joint condition 

    Args:
        s_codec : (B, N, 4)
        c_codec : (B, N, 7)
    """

    return torch.cat((s_codec, c_codec), dim=2)

def apply_normalization(cd, mean, std, i):
    # apply normalization for p codec in codec data
    cd['p_codec'][:, i] = (cd['p_codec'][:, i] - mean) / std

    return cd

def dataset_normalization(codec_data):
    """ normalize the p_codec, across the dataset range. 
        return mean and std for each column. 
    """
    dataset_pc = np.vstack([cd['p_codec'] for cd in codec_data])

    means, stds = [], []
    for i in range(5):
        mean = dataset_pc[:, i].mean() 
        std = dataset_pc[:, i].std() 
        codec_data = list(map(apply_normalization, codec_data, 
                              [mean] * len(codec_data), 
                              [std] * len(codec_data),
                              [i] * len(codec_data)))
        means.append(float(mean))
        stds.append(float(std))  # conversion for save into OmegaConf

    return codec_data, means, stds

def p_codec_scale(p_codec, means, stds):
    # inverse of normalization applied on p_codec 
    # p_codec: (B, N, 5) or (B, 1, N, 5)
    for i in range(5):
        p_codec[..., i] = p_codec[..., i] * stds[i] + means[i]

    return p_codec


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