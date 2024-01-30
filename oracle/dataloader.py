import os, sys, glob
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict, Counter
from omegaconf import OmegaConf
from tqdm import tqdm

from matrix import perfmidi_to_matrix
import hook


ASAP_DIR = "/homes/hz009/Research/Datasets/asap-dataset-alignment"



class MIDICepDataloader:
    def __init__(self, split_ratio=0.9, mode='train'):

        self.data_pair = self.create_correspondance()

        # Split the data into train and test
        total_pairs = len(self.data_pair)
        split_index = int(total_pairs * split_ratio)

        if mode == 'train':
            self.data_pair = self.data_pair[:split_index]
        elif mode == 'test':
            self.data_pair = self.data_pair[split_index:]


    def create_correspondance(self, trial=False):

        if trial:
            data_pair = np.load("test_data.npy", allow_pickle=True)

        else:
            performance_paths = glob.glob(os.path.join(ASAP_DIR, "**/*[!e].mid"), recursive=True)
            alignment_paths = [(pp[:-4] + "_note_alignments/note_alignment.tsv") for pp in performance_paths]
            score_paths = [os.path.join("/".join(pp.split("/")[:-1]), "xml_score.musicxml") for pp in performance_paths]
            cep_feat_paths = [pp[:-4] + "_cep_features.csv" for pp in performance_paths]

            data_pair =  []
            for s_path, p_path, a_path, c_path in tqdm(zip(score_paths, performance_paths, alignment_paths, cep_feat_paths)):
                if (os.path.exists(s_path) and os.path.exists(a_path)):
                        
                    piece_name = "_".join(s_path.split("alignment/")[-1].split("/")[:-1])
                    if os.path.exists(c_path): 
                        
                        convert_cfg = OmegaConf.create({
                            "segmentation": {
                                "seg_type": 'fix_time',
                                "seg_time": 15,
                                "seg_hop": 5
                            },
                            'matrix':{
                                "resolution": 800,
                                "bins": 131
                            },
                            "experiment": {
                                "feat_level": 1
                            }
                        })
                        matrix = perfmidi_to_matrix(p_path, convert_cfg)

                        cep_feats = np.array(pd.read_csv(c_path))[:, 1:] # first column is time
                        data_pair.extend(zip(matrix, cep_feats))

        return data_pair



    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        matrix, cep = self.data_pair[idx]
        
        return {
            "matrix": matrix,
            "cep": cep
        }



if __name__ == "__main__":

    # Create dataloader
    dataloader = MIDICepDataloader()

    # Get pairs
    pairs = dataloader.data_pair
    hook()

    # Example usage: print first 5 pairs
    for pair in pairs[:5]:
        print(pair)
