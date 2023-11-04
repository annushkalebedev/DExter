import os, sys, glob, copy
import torch
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
from utils import *
import hook



class Renderer():
    """Renderer class that takes in one codec sample and convert to MIDI, and analyze.

    - sampled_pcodec:  (B, L, 4)
    - source_data: dictionary contains {score, snote_id, piece_name...}
    - label_data: same with source data

    """
    def __init__(self, 
                 sampled_pcodec=None,  
                 source_data=None, label_data=None,
                 save_path=None, 
                 with_source=False,
                 means=None, stds=None,
                 idx=0, B=16):
        
        self.sampled_pcodec = sampled_pcodec
        self.source_data = source_data
        self.label_data = label_data
        self.save_path = save_path
        self.with_source = with_source
        self.means = means
        self.stds = stds
        self.idx = idx
        self.B = B

    def load_external_performances(self, performance_path, score_path, snote_ids):
        """load the performance that's already generated (for evaluating other models)"""

        self.performed_part = pt.load_performance(performance_path).performedparts[0]
        self.score = pt.load_score(score_path)
        self.snote_ids = snote_ids

        return 

    def render_sample(self, save_sourcelabel):
        """render the sample to midi file and save 
        """

        self.performance_array = self.parameters_to_performance_array(self.sampled_pcodec)
        self.pcodec_label = self.parameters_to_performance_array(self.label_data['p_codec'].cpu())
        self.pcodec_source = self.parameters_to_performance_array(self.source_data['p_codec'].cpu())

        try:
            # load the batch information and decode into performed parts
            self.load_and_decode()

            # compute the tempo curve of sampled parameters (avg for joint-onsets)
            self.compare_performance_curve()

            # get loss and correlation metrics
            tempo_vel_loss = F.l1_loss(torch.tensor(self.performed_tempo), torch.tensor(self.label_tempo)) + \
                                    F.l1_loss(torch.tensor(self.performed_vel), torch.tensor(self.label_vel))
            tempo_vel_cor = pearsonr(self.performed_tempo, self.label_tempo)[0] + pearsonr(self.performed_vel, self.label_vel)[0]

            pt.save_performance_midi(self.performed_part, f"{self.save_path}/{self.idx}_{self.piece_name}.mid")
            if save_sourcelabel:
                pt.save_performance_midi(self.performed_part_label, f"{self.save_path}/{self.idx}_{self.piece_name}_label.mid")
                pt.save_performance_midi(self.performed_part_source, f"{self.save_path}/{self.idx}_{self.piece_name}_source.mid")
        except Exception as e:
            print(e)

        return tempo_vel_loss, tempo_vel_cor


    def load_and_decode(self, performance_array, pcodec_label, source_data):
        """load the meta information (scores, snote_ids and piece name)
        then decode the p_codecs into performed parts  
         """
        # update the snote_id_path to the new one
        snote_id_path = source_data['snote_id_path']
        self.snote_ids = np.load(snote_id_path)

        if len(self.snote_ids) < 10: # when there is too few notes, the rendering would have problems.
            raise RuntimeError("snote_ids too short")
        self.score = pt.load_musicxml(source_data['score_path'], force_note_ids='keep')
        # unfold the score if necessary (mostly for ASAP)
        if ("-" in self.snote_ids[0] and 
            "-" not in score.note_array()['id'][0]):
            score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
        self.piece_name = source_data['piece_name']
        N = len(self.snote_ids)
        
        pad_mask = np.full(self.snote_ids.shape, False)
        self.performed_part = pt.musicanalysis.decode_performance(score, performance_array[:N], snote_ids=self.snote_ids, pad_mask=pad_mask)
        self.performed_part_label = pt.musicanalysis.decode_performance(score, pcodec_label[:N], snote_ids=self.snote_ids, pad_mask=pad_mask)
        self.performed_part_source = pt.musicanalysis.decode_performance(score, source_data['p_codec'][:N], snote_ids=self.snote_ids, pad_mask=pad_mask)


    def compare_performance_curve(self):
        """compute the performance curve (tempo curve \ velocity curve) from given performance array
            pcodec_original: another parameter curve, coming from the optional starting point of transfer

        Returns: (mark for self)
            onset_beats : 
            performed_tempo
            label_tempo
            source_tempo
            performed_vel
            label_vel
            source_vel 
        """
        na = self.score.note_array()
        na = na[np.in1d(na['id'], self.snote_ids)]

        pcodecs = [self.pcodec_pred, self.pcodec_label, self.pcodec_source]

        self.onset_beats = np.unique(na['onset_beat'])
        res = [self.onset_beats]
        for pcodec in pcodecs:

            joint_pcodec = rfn.merge_arrays([na, pcodec[:len(na)]], flatten = True, usemask = False)
            bp = [joint_pcodec[joint_pcodec['onset_beat'] == ob]['beat_period'].mean() for ob in self.onset_beats]
            vel = [joint_pcodec[joint_pcodec['onset_beat'] == ob]['velocity'].mean() for ob in self.onset_beats]
            tempo_curve_pred = interp1d(self.onset_beats, 60 / np.array(bp))
            tempo_curve, velocity_curve = 60 / np.array(bp), np.array(vel)
            if np.isinf(tempo_curve).any():
                raise RuntimeError("inf in tempo")
            res.extend([tempo_curve, velocity_curve])

        _, self.performed_tempo, self.performed_vel, self.label_tempo, self.label_vel, self.source_tempo, self.source_vel = res


    def plot_curves(self, ax):
        ax.flatten()[self.idx].plot(self.beats, self.performed_tempo, label="performed_tempo")
        ax.flatten()[self.idx].plot(self.beats, self.label_tempo, label="label_tempo")
        ax.flatten()[self.idx].set_ylim(0, 300)

        ax.flatten()[self.idx+self.B].plot(self.beats, self.performed_vel, label="performed_vel")
        ax.flatten()[self.idx+self.B].plot(self.beats, self.label_vel, label="label_vel")

        if self.with_source:
            ax.flatten()[self.idx].plot(self.beats, self.source_tempo, label="source_tempo")
            ax.flatten()[self.idx+self.B].plot(self.beats, self.source_vel, label="source_vel")

        ax.flatten()[self.idx].legend()
        ax.flatten()[self.idx].set_title(f"tempo: {self.piece_name}")   
        ax.flatten()[self.idx+self.B].legend()
        ax.flatten()[self.idx+self.B].set_title(f"vel: {self.piece_name}")


    def parameters_to_performance_array(self, parameters):
        """
            parameters (np.ndarray) : shape (B, N, 5)
        """
        parameters = list(zip(*parameters.T))
        performance_array = np.array(parameters, 
                            dtype=[("beat_period", "f4"), ("velocity", "f4"), ("timing", "f4"), ("articulation_log", "f4"), ("pedal", "f4")])

        return performance_array



    def quantitative_analysis(self, eval_save_path):
        '''
        For codec attributes:
            - Tempo and velocity deviation
            - Tempo and velocity correlation
        
        For performance features:
            - Deviation of each parameter on note level (for articulation: drop the ones with mask?)
            - Distribution difference of each parameters using KL estimation.
        '''
        alignment = [{'label': "match", "score_id": sid, "performance_id": sid} for sid in self.snote_ids]

        self.feats_pred, res = pt.musicanalysis.compute_performance_features(self.score, self.performed_part, alignment, feature_functions='all')
        self.feats_pred.to_csv(f"{eval_save_path}_feats_pred.csv", index=False)

        
        if not type(self.source_data) == type(None):
            self.feats_label, res = pt.musicanalysis.compute_performance_features(self.score, self.performed_part_label, alignment, feature_functions='all')
            self.feats_source, res = pt.musicanalysis.compute_performance_features(self.score, self.performed_part_source, alignment, feature_functions='all')

            self.feats_label.to_csv(f"{eval_save_path}_feats_label.csv", index=False)
            self.feats_source.to_csv(f"{eval_save_path}_feats_source.csv", index=False)
        
        return {
            "feats_pred": self.feats_pred,
            "feats_label": self.feats_label,
            "feats_source": self.feats_source,
        }

    def distribution_analysis(self):
        features_distribution = {}
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
            features_distribution[feat_name] = dev_kl_cor_estimate(self.feats_pred[feat_name], self.feats_label[feat_name], self.feats_source[feat_name],
                                                            mask=mask)

        features_distribution["tempo_curve"] = dev_kl_cor_estimate(self.performed_tempo, self.label_tempo, self.source_tempo, mask=np.full(self.performed_tempo.shape, False))    
        features_distribution["vel_curve"] = dev_kl_cor_estimate(self.performed_vel, self.label_vel, self.source_vel, mask=np.full(self.performed_vel.shape, False))    
        self.features_distribution = pd.DataFrame(features_distribution)


    def dev_kl_cor_estimate(self, pred_feat, label_feat, source_feat, 
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