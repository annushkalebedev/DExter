import os, sys, glob, copy
import torch
from collections import defaultdict
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, gaussian_kde, entropy
from scipy.special import kl_div
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from prepare_data import *
import hook


def tensor_pair_swap(x):
    if type(x) == list:
        x = np.array(x)
    # given batched x, swap the pairs from the first dimension
    permute_index = torch.arange(x.shape[0]).view(-1, 2)[:, [1, 0]].contiguous().view(-1)
    return x[permute_index]


def get_batch_slice(batch, idx):
    """given a dictionary batch, get the sliced ditionary given idx"""

    return {k: v[idx] for k, v in batch.items()}


def render_sample(sampled_parameters,  batch_source, batch_label, save_path, 
                  with_source=False, evaluate=False,
                  means=None, stds=None):
    """
    render the sample to midi file and save 
        sampled_parameters (np.array) : (B, L, 4)
        batch : dictionary
            "score_path" : score_path to load
            "snote_id_path" : snote_id path to load
        save_path: root directory to save
        with_source: whether to use the source.
        save_interpolation: weather 
        
    """
    B = len(sampled_parameters) 

    # rescale the predictions and labels back to normal
    sampled_parameters = p_codec_scale(sampled_parameters, means, stds)
    batch_source['p_codec'] = p_codec_scale(batch_source['p_codec'], means, stds)
    batch_label['p_codec'] = p_codec_scale(batch_label['p_codec'], means, stds)

    fig, ax = plt.subplots(int(B/2), 4, figsize=(24, 3*B))
    eval_results = []

    for idx in range(B): 
        
        performance_array = parameters_to_performance_array(sampled_parameters[idx])
        pcodec_label = parameters_to_performance_array(batch_label['p_codec'][idx].cpu())
        pcodec_source = parameters_to_performance_array(batch_source['p_codec'][idx].cpu())

        try:
            # load the batch information and decode into performed parts
            (performed_part, performed_part_label, performed_part_source, score, piece_name, snote_ids) = load_and_decode(
                performance_array, pcodec_label, pcodec_source, batch_source, idx)

            # compute the tempo curve of sampled parameters (avg for joint-onsets)
            beats, performed_tempo, performed_vel, label_tempo, label_vel, source_tempo, source_vel = compare_performance_curve(
                                                        score, snote_ids, performance_array, pcodec_label, pcodec_source=pcodec_source)

            # get loss and correlation metrics
            tempo_vel_loss = F.l1_loss(torch.tensor(performed_tempo), torch.tensor(label_tempo)) + \
                                    F.l1_loss(torch.tensor(performed_vel), torch.tensor(label_vel))
            tempo_vel_cor = pearsonr(performed_tempo, label_tempo)[0] + pearsonr(performed_vel, label_vel)[0]

            if evaluate: # provide analysis to the generated performance 
                eval_res = quantitative_analysis(score, snote_ids, performed_part, performed_part_label, performed_part_source,
                                                performed_tempo, label_tempo, source_tempo, performed_vel, label_vel, source_vel)
                eval_results.append(eval_res['features_results'])

            plot_curves(ax, idx, B, beats, performed_tempo, label_tempo, performed_vel, label_vel, source_tempo, source_vel, piece_name, with_source)

            pt.save_performance_midi(performed_part, f"{save_path}/{idx}_{piece_name}.mid")
            if evaluate:
                pt.save_performance_midi(performed_part_label, f"{save_path}/{idx}_{piece_name}_label.mid")
                pt.save_performance_midi(performed_part_source, f"{save_path}/{idx}_{piece_name}_source.mid")
        except Exception as e:
            print(e)

    return fig, tempo_vel_loss, tempo_vel_cor, eval_results

def load_and_decode(performance_array, pcodec_label, pcodec_source, batch_source, idx):
    # update the snote_id_path to the new one
    # snote_id_path = batch['snote_id_path'][idx].replace("data", "/import/c4dm-datasets-ext/DiffPerformer")
    snote_id_path = batch_source['snote_id_path'][idx]
    snote_ids = np.load(snote_id_path)

    if len(snote_ids) < 10: # when there is too few notes, the rendering would have problems.
        raise RuntimeError("snote_ids too short")
    score = pt.load_musicxml(batch_source['score_path'][idx], force_note_ids='keep')
    # unfold the score if necessary (mostly for ASAP)
    if ("-" in snote_ids[0] and 
        "-" not in score.note_array()['id'][0]):
        score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    piece_name = batch_source['piece_name'][idx]
    N = len(snote_ids)
    
    pad_mask = np.full(snote_ids.shape, False)
    performed_part = pt.musicanalysis.decode_performance(score, performance_array[:N], snote_ids=snote_ids, pad_mask=pad_mask)
    performed_part_label = pt.musicanalysis.decode_performance(score, pcodec_label[:N], snote_ids=snote_ids, pad_mask=pad_mask)
    performed_part_source = pt.musicanalysis.decode_performance(score, pcodec_source[:N], snote_ids=snote_ids, pad_mask=pad_mask)

    return performed_part, performed_part_label, performed_part_source, score, piece_name, snote_ids


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
        tempo_curve, velocity_curve = 60 / np.array(bp), np.array(vel)
        if np.isinf(tempo_curve).any():
            raise RuntimeError("inf in tempo")
        res.extend([tempo_curve, velocity_curve])

    return res

def plot_curves(ax, idx, B, beats, performed_tempo, label_tempo, performed_vel, label_vel, source_tempo, source_vel, piece_name, with_source=False):
    ax.flatten()[idx].plot(beats, performed_tempo, label="performed_tempo")
    ax.flatten()[idx].plot(beats, label_tempo, label="label_tempo")
    ax.flatten()[idx].set_ylim(0, 300)

    ax.flatten()[idx+B].plot(beats, performed_vel, label="performed_vel")
    ax.flatten()[idx+B].plot(beats, label_vel, label="label_vel")

    if with_source:
        ax.flatten()[idx].plot(beats, source_tempo, label="source_tempo")
        ax.flatten()[idx+B].plot(beats, source_vel, label="source_vel")

    ax.flatten()[idx].legend()
    ax.flatten()[idx].set_title(f"tempo: {piece_name}")   
    ax.flatten()[idx+B].legend()
    ax.flatten()[idx+B].set_title(f"vel: {piece_name}")
    return 


def parameters_to_performance_array(parameters):
    """
        parameters (np.ndarray) : shape (B, N, 5)
    """
    parameters = list(zip(*parameters.T))
    performance_array = np.array(parameters, 
                        dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4"), ("pedal", "f4")])

    return performance_array


def quantitative_analysis(score, snote_ids, alignment, performed_part, performed_part_label, performed_part_source,
                          performed_tempo, label_tempo, source_tempo, performed_vel, label_vel, source_vel, eval_save_path):
    '''
    For codec attributes:
        - Tempo and velocity deviation
        - Tempo and velocity correlation
    
    For performance features:
        - Deviation of each parameter on note level (for articulation: drop the ones with mask?)
        - Distribution difference of each parameters using KL estimation.
    '''
    alignment = [{'label': "match", "score_id": sid, "performance_id": sid} for sid in snote_ids]

    feats_pred, res = pt.musicanalysis.compute_performance_features(score, performed_part, alignment, feature_functions='all')
    feats_label, res = pt.musicanalysis.compute_performance_features(score, performed_part_label, alignment, feature_functions='all')
    feats_source, res = pt.musicanalysis.compute_performance_features(score, performed_part_source, alignment, feature_functions='all')

    feats_pred.to_csv(f"{eval_save_path}_feats_pred.csv", index=False)
    feats_pred.to_csv(f"{eval_save_path}_feats_pred.csv", index=False)
    feats_pred.to_csv(f"{eval_save_path}_feats_pred.csv", index=False)
    
    features_results = {}
    for feat_name in ['articulation_feature.kor',
                      'asynchrony_feature.pitch_cor',
                      'asynchrony_feature.vel_cor',
                      'asynchrony_feature.delta',
                      'dynamics_feature.agreement',
                      'dynamics_feature.consistency_std',
                      'dynamics_feature.ramp_cor',
                      'dynamics_feature.tempo_cor',
                      'pedal_feature.onset_value'
                      ]:
        
        mask = np.full(res['no_kor_mask'].shape, False)
        if 'kor' in feat_name:
            mask = res['no_kor_mask']
        features_results[feat_name] = dev_kl_cor_estimate(feats_pred[feat_name], feats_label[feat_name], feats_source[feat_name],
                                                        mask=mask)

    features_results["tempo_curve"] = dev_kl_cor_estimate(performed_tempo, label_tempo, source_tempo, mask=np.full(performed_tempo.shape, False))    
    features_results["vel_curve"] = dev_kl_cor_estimate(performed_vel, label_vel, source_vel, mask=np.full(performed_vel.shape, False))    
    features_results = pd.DataFrame(features_results)

    return {
        "feats_pred": feats_pred,
        "feats_label": feats_label,
        "feats_source": feats_source,
        "features_results": features_results
    }


def dev_kl_cor_estimate(pred_feat, label_feat, source_feat, 
                        N=300, mask=None):
    """
        dev: deviation between the prediction and the target / source we want to compare with.
        KL: MC estimate KL divergence by convering the features into kde distribution, and sample their pdf
        cor: correlation between the two compared series

        mask: for the values that we don't want to look at. 
    """
    pred_feat, label_feat, source_feat = pred_feat[~mask], label_feat[~mask], source_feat[~mask]

    try:
        kde_pred = gaussian_kde(pred_feat)
        kde_label = gaussian_kde(label_feat)
        kde_source = gaussian_kde(source_feat)

        pred_points = kde_pred.resample(N) 
        label_points = kde_label.resample(N) 
        pl_KL = entropy(kde_pred.pdf(pred_points), kde_label.pdf(pred_points))
        ps_KL = entropy(kde_pred.pdf(pred_points), kde_source.pdf(pred_points))
        ls_KL = entropy(kde_label.pdf(label_points), kde_source.pdf(label_points))
    except Exception as e:
        pl_KL, ps_KL, ls_KL = -1, -1, -1                                                                                                                                                                                                                                                                                        
    
    return {
        "label_dev(percent)": np.ma.masked_invalid((pred_feat - label_feat) / label_feat).mean(), # filter out the inf and nan
        "source_dev(percent)": np.ma.masked_invalid((pred_feat - source_feat) / (source_feat)).mean(),
        "pred-label(KL)": pl_KL,
        "pred-source(KL)": ps_KL,
        "label-source(KL)": ls_KL,
        "pred-label(cor)": pearsonr(pred_feat, label_feat)[0],
        "pred-source(cor)": pearsonr(pred_feat, source_feat)[0],
        "label-source(cor)": pearsonr(label_feat, source_feat)[0],
    }


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

def split_train_valid(codec_data, select_num=58008):
    """select the train and valid set according to the following criteria: 
        - ASAP and VIENNA422 goes to testing set as they are better ground truths
        - split regarding to pieces.
    """

    train_idx = int(len(codec_data) * 0.85) - 1
    assert (train_idx % 2 == 0)
    if not select_num:
        return codec_data[:train_idx], codec_data[train_idx:]

    selected_cd, unselected_cd = [], []
    for idx in range(0, len(codec_data), 2):
        if len(selected_cd) > select_num:  
            break
        cd, cd_ = codec_data[idx], codec_data[idx+1]
        if "ATEPP" not in cd['snote_id_path']:
            selected_cd.extend([cd, cd_])
        else:
            unselected_cd.extend([cd, cd_])
    
    # selected_cd, unselected_cd = defaultdict(list), []
    # for cd in codec_data:
    #     for name in TESTING_GROUP:
    #         if (not selected_cd[name]) and name in cd['snote_id_path']:
    #             selected_cd[name] = cd
    #             break
    #     else:
    #         unselected_cd.append(cd)
            
    
    # np.random.shuffle(unselected_cd)
    train_set = unselected_cd[:train_idx]
    valid_set = selected_cd + unselected_cd[train_idx:]

    return train_set, valid_set

def load_transfer_pair(K=50000, N=200):
    """
        returns:
        - transfer_pairs: list of tuple of codec
        - unpaired: list of single codec
    """
    transfer_pairs = np.load(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy", allow_pickle=True)
    unpaired = np.load(f"{BASE_DIR}/codec_N={N}_mixup_unpaired_K={K}.npy", allow_pickle=True)
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

def apply_normalization(cd, mean, std, i, idx):
    # apply normalization for p codec in codec data
    return (cd['p_codec'][:, i] - mean) / std

def dataset_normalization(train_set, valid_set):
    """ normalize the p_codec, across the dataset range. 
        return mean and std for each column. 
    """
    codec_data = np.hstack([train_set, valid_set])
    dataset_pc = np.vstack([cd['p_codec'] for cd in codec_data])
    means, stds = [], []
    codec_data_ = copy.deepcopy(codec_data)
    for i in range(5):
        mean = dataset_pc[:, i].mean() 
        std = dataset_pc[:, i].std() 
        for idx, cd in enumerate(codec_data):
            codec_data_[idx]['p_codec'][:, i] = apply_normalization(cd, mean, std, i, idx)
        means.append(float(mean))
        stds.append(float(std))  # conversion for save into OmegaConf

    return codec_data_[:len(train_set)], codec_data_[len(train_set):], means, stds

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


def render_midi_to_audio(midi_path, output_path=None):
    """The soundfont we used is the Essential Keys-sofrzando-v9.6 from https://sites.google.com/site/soundfonts4u/ """

    if output_path == None:
        output_path = midi_path[:-4] + ".wav"

    os.system(f"fluidsynth -ni ../artifacts/Essential-Keys-sforzando-v9.6.sf2 {midi_path} -F {output_path} ")
    return



if __name__ == "__main__":

    import random
    asap_mid = glob.glob("../Datasets/asap-dataset-alignment/**/*.mid", recursive=True)
    for am in random.choices(asap_mid, k=2):
        render_midi_to_audio(am, output_path=f"{am[35:-4]}.wav")
    hook()

    pass