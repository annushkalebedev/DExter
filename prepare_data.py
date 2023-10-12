import os, sys, glob
import argparse
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt

import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import hook


VIENNA_MATCH_DIR = "../Datasets/vienna4x22/match/"
VIENNA_MUSICXML_DIR = "../Datasets/vienna4x22/musicxml/"
VIENNA_PERFORMANCE_DIR = "../Datasets/vienna4x22/midi/"

ASAP_DIR = "../Datasets/asap-dataset-alignment/"

BMZ_MATCH_DIR = "../Datasets/pianodata-master/match"
BMZ_MUSICXML_DIR = "../Datasets/pianodata-master/xml/"

ATEPP_DIR = "../Datasets/ATEPP-1.1"
ATEPP_META_DIR = "../Datasets/ATEPP-1.1/ATEPP-metadata-1.3.csv"

BASE_DIR = "/import/c4dm-datasets-ext/DiffPerformer"


def load_dataset_codec(dataset='ASAP', return_metadata=False):
    """load the performance features for the given dataset from the saved
    .npy arrays. Return list of numpy arrays.

    Args:
        dataset (str): dataset to process. Defaults to 'ASAP'.
        return_metadata (bool): return a dict with the list of composer and performer of the features. optional.  
    """

    if dataset == "VIENNA422":
        pf_paths = glob.glob(os.path.join(VIENNA_MATCH_DIR, "*.npy"))
        meta_dict = {"composer": [path.split("/")[4].split("_")[0] for path in pf_paths], 
                     "performer": [path.split("/")[4].split("_")[-3] for path in pf_paths]}
    if dataset == "ASAP":
        pf_paths = glob.glob(os.path.join(ASAP_DIR, "**/*.npy"), recursive=True)
        meta_dict = {"composer": [path.split("/")[3] for path in pf_paths], 
                     "performer": [path.split("/")[-1].split("_")[0] for path in pf_paths]}
    if dataset == "ATEPP":
        pf_paths = glob.glob(os.path.join(ATEPP_DIR, "**/*.npy"), recursive=True)
        meta_csv = pd.read_csv(ATEPP_META_DIR)
        meta_dict = {"composer": [path.split("/")[3] for path in pf_paths], 
                     "performer": [meta_csv[meta_csv['midi_path'].str.contains(
                                        path.split("/")[-1][:5])]['artist'].item() for path in pf_paths]}
    if dataset == "BMZ":
        pf_paths = glob.glob(os.path.join(BMZ_MATCH_DIR, "**/*.npy"), recursive=True)
        composer = [path.split("/")[-1].split("_")[0] for path in pf_paths]
        meta_dict = {"composer": composer, 
                     "performer": ["Magdaloff" if c == "chopin" else "Zeillinger" for c in composer]}

    pf = [np.load(path) for path in pf_paths]
    if return_metadata:
        return pf, meta_dict
    return pf


def process_dataset_codec(max_note_len, mix_up=False):
    """process the performance features for the given dataset. Save the 
    computed features in the form of numpy arrays in the same directory as 
    performance data.

    Args:
        dataset (str): dataset to process. Defaults to 'ASAP'.
    """

    prev_s_path, data = None, []
    os.makedirs(f"{BASE_DIR}/snote_ids/N={max_note_len}", exist_ok=True)
    for dataset in [
                    'ASAP', 
                    'VIENNA422',
                    'ATEPP'
                    ]:

        if dataset == "VIENNA422":
            alignment_paths = glob.glob(os.path.join(VIENNA_MATCH_DIR, "*[!e].match"))
            alignment_paths = sorted(alignment_paths)
            score_paths = [(VIENNA_MUSICXML_DIR + pp.split("/")[-1][:-10] + ".musicxml") for pp in alignment_paths]
            performance_paths = [None] * len(alignment_paths) # don't use the given performance, use the aligned.
            cep_feat_paths = [("../Datasets/vienna4x22/cep_features/" + pp.split("/")[-1][:-6] + ".csv") for pp in alignment_paths]
        if dataset == "ASAP":
            performance_paths = glob.glob(os.path.join(ASAP_DIR, "**/*[!e].mid"), recursive=True)
            alignment_paths = [(pp[:-4] + "_note_alignments/note_alignment.tsv") for pp in performance_paths]
            score_paths = [os.path.join("/".join(pp.split("/")[:-1]), "xml_score.musicxml") for pp in performance_paths]
            cep_feat_paths = [pp[:-4] + "_cep_features.csv" for pp in performance_paths]
        if dataset == "ATEPP":
            alignment_paths = glob.glob(os.path.join(ATEPP_DIR, "**/[!z]*n.csv"), recursive=True)
            alignment_paths = sorted(alignment_paths)
            performance_paths = [(aa[:-10] + ".mid") for aa in alignment_paths]
            score_paths = [glob.glob(os.path.join("/".join(pp.split("/")[:-1]), "*.*l"))[0] for idx, pp in enumerate(performance_paths)]
            cep_feat_paths = ["/".join(aa.split("/")[:-1]) + "_cep_features.csv" for aa in alignment_paths]
            atepp_overlap_dirs = get_atepp_overlap()
            atepp_overlap_dirs = [f"{ATEPP_DIR}/{ao_dir}" for ao_dir in atepp_overlap_dirs]

        # storing codecs for existing score
        same_score_p_codec, mixuped_p_codec = [], []
        same_score_c_codec, mixuped_c_codec = [], []
        for s_path, p_path, a_path, c_path in tqdm(zip(score_paths, performance_paths, alignment_paths, cep_feat_paths)):

            # parsing error
            if s_path == "../Datasets/pianodata-master/xml/chopin_op35_Mv3.xml": # BMZ
                continue
            if s_path == '../Datasets/asap-dataset-alignment/Chopin/Scherzos/31/xml_score.musicxml': # ASAP
                continue
            if s_path == '../Datasets/asap-dataset-alignment/Ravel/Gaspard_de_la_Nuit/1_Ondine/xml_score.musicxml': # ASAP
                continue
            if a_path == '../Datasets/asap-dataset-alignment/Beethoven/Piano_Sonatas/23-1/LiuC02M_note_alignments/note_alignment.tsv':
                continue # tempo_grad floating point error
            if s_path == '../Datasets/asap-dataset-alignment/Scriabin/Sonatas/5/xml_score.musicxml': # ASAP tempo
                continue
            if s_path == '../Datasets/asap-dataset-alignment/Chopin/Etudes_op_25/2/xml_score.musicxml':
                continue
            if s_path == '../Datasets/ATEPP-1.1/Frederic_Chopin/Nocturne_No.13_in_C_minor,_Op._48_No._1/score.xml':
                continue
            if s_path == '../Datasets/ATEPP-1.1/Frederic_Chopin/24_Preludes,_Op._28/No._7_in_A_Major:_Andantino/Prlude_Opus_28_No._7_in_A_Major.mxl':
                continue

            if (os.path.exists(s_path) and os.path.exists(a_path)):

                if dataset == 'ATEPP': # ATEPP: skip the bad ones
                    alignment = pt.io.importparangonada.load_parangonada_alignment(a_path)
                    match_aligns = [a for a in alignment if a['label'] == 'match']
                    insertion_aligns = [a for a in alignment if a['label'] == 'insertion']
                    deletion_aligns = [a for a in alignment if a['label'] == 'deletion']
                    if (len(match_aligns) / (len(insertion_aligns) + len(deletion_aligns) + len(match_aligns))) < 0.5:
                        continue

                # path to save the reproducing artifacts
                if dataset == "VIENNA422":
                    piece_name = s_path.split("/")[-1].split(".")[0]
                    
                if dataset == "ASAP":
                    piece_name = "_".join(s_path.split("alignment/")[-1].split("/")[:-1])
                if dataset == 'ATEPP':
                    piece_name = p_path.split("/")[-1][:-4] 
                    perf_name = p_path.split("/")[-1][:-4] 
                save_snote_id_path = f"{BASE_DIR}/snote_ids/N={max_note_len}/{dataset}_{piece_name}"

                # encode!
                if prev_s_path == s_path:
                    p_codec, score, snote_ids, m_score = get_performance_codec(s_path, a_path, p_path, score=score)
                    p_codec = rfn.structured_to_unstructured(p_codec)

                    # c_codec = np.full((len(p_codec), 7), 0) # placeholder
                    if os.path.exists(c_path): # c codec
                        cep_feats = pd.read_csv(c_path)
                        c_codec = rfn.structured_to_unstructured(get_cep_codec(cep_feats, m_score))
                    else:
                        continue

                    if 60 / p_codec[:, 0].mean() > 200: # filter tempo
                        continue

                    # get the mixuped codec with prev performances with the same score
                    if mix_up and ((dataset == "VIENNA422") or (dataset == 'ATEPP' and "/".join(s_path.split("/")[:-1]) in atepp_overlap_dirs)): 
                        mixuped_p_codec = [np.mean( np.array([ p_codec, ss_p_codec ]), axis=0 ) for ss_p_codec in same_score_p_codec]
                        same_score_p_codec.append(p_codec)
                        mixuped_c_codec = [np.mean( np.array([ c_codec, ss_c_codec ]), axis=0 ) for ss_c_codec in same_score_c_codec]
                        same_score_c_codec.append(c_codec)

                else:
                    p_codec, score, snote_ids, m_score = get_performance_codec(s_path, a_path, p_path)
                    p_codec = rfn.structured_to_unstructured(p_codec)
                    same_score_p_codec, mixuped_p_codec = [], []
                    same_score_c_codec, mixuped_c_codec = [], []

                    if 60 / p_codec[:, 0].mean() > 200: # filter tempo
                        continue

                    # c_codec = np.full((len(p_codec), 7), 0)  # placeholder
                    if os.path.exists(c_path): # c codec
                        cep_feats = pd.read_csv(c_path)
                        c_codec = rfn.structured_to_unstructured(get_cep_codec(cep_feats, m_score))
                    else:
                        print(f"no cep_features for {a_path}")
                        continue

                sna = score.note_array()
                sna = sna[np.in1d(sna['id'], snote_ids)]
                s_codec = rfn.structured_to_unstructured(
                    sna[['onset_div', 'duration_div', 'pitch', 'voice']])

                if not ((len(p_codec) == len(s_codec)) and (len(p_codec) == len(snote_ids))):
                    print(f"{a_path} has length issue: p: {len(p_codec)}; s: {len(s_codec)}") 
                    continue

                for i, (p_codec, c_codec) in enumerate(zip(([p_codec] + mixuped_p_codec), ([c_codec] + mixuped_c_codec))): # segmentation
                    if i == 1:
                        piece_name = piece_name + "_mixup"  # mixuped codec name

                    for idx in range(0, len(p_codec), max_note_len): # split the piece 
                        seg_p_codec = p_codec[idx : idx + max_note_len]
                        seg_s_codec = s_codec[idx : idx + max_note_len]
                        seg_c_codec = c_codec[idx : idx + max_note_len]
                        seg_snote_ids = snote_ids[idx : idx + max_note_len]

                        if len(seg_p_codec) < max_note_len:
                            seg_p_codec = np.pad(seg_p_codec, ((0, max_note_len - len(seg_p_codec)), (0, 0)), mode='constant', constant_values=0)
                            seg_s_codec = np.pad(seg_s_codec, ((0, max_note_len - len(seg_s_codec)), (0, 0)), mode='constant', constant_values=0)
                            seg_c_codec = np.pad(seg_c_codec, ((0, max_note_len - len(seg_c_codec)), (0, 0)), mode='constant', constant_values=0)

                        if len(seg_snote_ids) == 0:
                            hook()

                        seg_id_path = f"{save_snote_id_path}_seg{int(idx/max_note_len)}.npy"
                        # save snote_id
                        np.save(seg_id_path, seg_snote_ids) 

                        data.append({
                                    "p_codec": seg_p_codec, 
                                    "s_codec": seg_s_codec,
                                    "c_codec": seg_c_codec,
                                    "snote_id_path": seg_id_path,
                                    "score_path": s_path,
                                    "piece_name": piece_name  # piece name for shortcut and identifying the generated sample. unique for performance not composition
                                    })

                prev_s_path = s_path
            else:
                print(f"Data incomplete for {a_path}")

    if mix_up:
        np.save(f"{BASE_DIR}/codec_N={max_note_len}_mixup_smoothed.npy", np.stack(data))
    else:
        np.save(f"{BASE_DIR}/codec_N={max_note_len}.npy", np.stack(data))

    return 


def get_performance_codec(score_path, alignment_path, performance_path=None, score=None):
    """compute the performance feature given score, alignment and performance path.
    """

    if isinstance(score, type(None)):
        score = pt.load_musicxml(score_path)

    if alignment_path[-5:] == "match":
        performance, alignment = pt.load_match(alignment_path)
    elif alignment_path[-3:] == "tsv":
        alignment = pt.io.importparangonada.load_alignment_from_ASAP(alignment_path)
    elif alignment_path[-3:] == "csv": # case for ATEPP
        alignment = pt.io.importparangonada.load_parangonada_alignment(alignment_path)
        score = pt.load_musicxml(score_path, force_note_ids='keep')


    # if doesn't match the note id in alignment, unfold the score.
    if (('score_id' in alignment[0]) 
        and ("-" in alignment[0]['score_id'])
        and ("-" not in score.note_array(include_divs_per_quarter=False)['id'][0])): 
        score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 

    if not isinstance(performance_path, type(None)): # use the performance if it's given
        performance = pt.load_performance(performance_path)

    # get the performance encodings
    parameters, snote_ids, pad_mask, m_score = pt.musicanalysis.encode_performance(score, performance, alignment, 
                                                                          tempo_smooth='derivative'
                                                                          )

    return parameters, score, snote_ids, m_score


def get_cep_codec(cep_feats, m_score):
    """ align the perceptual features with the performance, return
        an array in the same size of p_codec (structured array)
    """
    c_codec = pd.DataFrame()
    for row in m_score:
        next_window = cep_feats[cep_feats['frame_start_time'] <= row['p_onset']]
        if not len(next_window):
            c_codec = c_codec.append(cep_feats.iloc[-1])
        else:
            c_codec = c_codec.append(next_window.iloc[-1])

    records = c_codec.drop(columns=['frame_start_time']).to_records(index=False)
    c_codec = np.array(records, dtype = records.dtype.descr)
    return c_codec


def get_atepp_overlap(): 
    """get the atepp subset with pieces of more than 8+ performances
    Returns: 
    """
    atepp_meta = pd.read_csv(ATEPP_META_DIR)
    score_groups = atepp_meta.groupby(['score_path']).count().sort_values(['midi_path'], ascending=False)
    selected_scores = score_groups.iloc[:]
    selected_score_folders = ["/".join(score_entry.name.split("/")[:-1]) for _, score_entry in selected_scores.iterrows()]

    return selected_score_folders


def render_sample(score_part, sample_path, snote_ids_path):
    """render """
    snote_ids = np.load(snote_ids_path)
    for idx in range(32):
        performance_array = reverse_quantized_codec(np.load(sample_path)[idx])
        performed_part = pt.musicanalysis.decode_performance(score_part, performance_array, snote_ids=snote_ids)

        pt.save_performance_midi(performed_part, f"samples/sample_{idx}.mid")

    return performed_part


def codec_data_analysis():
    # look at the distribution of the bp, velocity...
    codec_data = np.load(f"{BASE_DIR}/codec_N=200.npy", allow_pickle=True) # (N_data, 1000, 4)
    p_codecs = [cd['p_codec'] for cd in codec_data]

    avg_tempo = [60 / pc[:, 0].mean() for pc in p_codecs]
    avg_tempo = [a for a in avg_tempo if a <= 500]
    plt.hist(avg_tempo, bins=100)
    plt.xlim((0, 500))
    plt.savefig("tmp.png")

    """
        - N=200 101,947 pieces of data. 
        - there are 673 with an avg_tempo > 1000 !! 2k with avg_tempo > 500. 75% of the data are under 200bpm 

    """

def plot_codec(data, ax0, ax1, ax2, fig):
    # plot the p_codec and s_codec, on the given two axes

    p_im = ax0.imshow(data["p_codec"].T, aspect='auto', origin='lower')
    ax0.set_yticks([0, 1, 2, 3, 4])
    ax0.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log", 'pedal'])
    fig.colorbar(p_im, orientation='vertical', ax=ax0)
    ax0.set_title(data['snote_id_path'])

    s_im = ax1.imshow(data['s_codec'].T, aspect='auto', origin='lower')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['onset_div', 'duration_div', 'pitch', 'voice'])
    fig.colorbar(s_im, orientation='vertical', ax=ax1)
    
    c_im = ax2.imshow(data['c_codec'].T, aspect='auto', origin='lower')
    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax2.set_yticklabels(['melodiousness', 'articulation', 'rhythm_complexity', 
                         'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness'])
    fig.colorbar(c_im, orientation='vertical', ax=ax2)

    return 


def plot_codec_list(codec_list):

    n_data = len(codec_list)
    fig, ax = plt.subplots(3 * n_data, 1, figsize=(24, 4 * n_data))
    for idx, data in enumerate(codec_list):
        plot_codec(data, ax[idx * 3], ax[idx * 3 + 1], ax[idx * 3 + 2], fig)

    plt.savefig("tmp.png")

    return fig

def match_midlevels():
    mid_paths = glob.glob("../Datasets/midlevel_2bf74_2/**/*.csv", recursive=True)
    for mp in mid_paths:
        newdir = "/".join(mp.split("/")[3:])[:-4]
        os.system(f"mv {mp} ../Datasets/asap-dataset-alignment/{newdir}_cep_features.csv")
    return 


def make_transfer_pair(codec_data, K=50000, N=200):
    """make transfer pair from the codec data for the testing set. 
        Transfer data come from real performance (not mix-up combinations), and the pieces used in testing set doesn't go into training. 

        returns:
        - transfer_pairs: list of tuple of codec
        - unpaired: list of single codec

        stats:
        - in total 2M pairs can be found. In testing we can use ~1000 pairs (K). rest can go into unpaired for training. For full transfer 
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
        if len(transfer_pairs) > K:
            break
        seg_id = cd['snote_id_path'].split("seg")[-1].split(".")[0]
        # find the one that belongs to the same piece, same segment number, but not itself
        mask = list(map(lambda x: ((
                                    x['score_path'] == cd['score_path']) 
                                   and (f"seg{seg_id}." in x['snote_id_path']) 
                                   and (x['p_codec'] != cd['p_codec']).any() 
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
    
    np.save(f"{BASE_DIR}/codec_N={N}_mixup_paired_K={K}.npy", transfer_pairs)
    np.save(f"{BASE_DIR}/codec_N={N}_mixup_unpaired_K={K}.npy", unpaired)
    
    return transfer_pairs, unpaired



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_codec', action='store_true')
    parser.add_argument('--pairing', action='store_true')
    parser.add_argument('--MAX_NOTE_LEN', type=int, default=200, required=False)
    parser.add_argument('--K', type=int, default=2000, required=False)
    args = parser.parse_args()

    if args.compute_codec:
        process_dataset_codec(args.MAX_NOTE_LEN, mix_up=True)
    elif args.pairing:
        codec_data = np.load(f"{BASE_DIR}/codec_N={args.MAX_NOTE_LEN}_mixup.npy", allow_pickle=True) 
        transfer_pairs, unpaired = make_transfer_pair(codec_data, K=args.K, N=args.MAX_NOTE_LEN)

    # codec_data_analysis()

    # from utils import parameters_to_performance_array

    # plot_codec_list(codec_data[:1])

    # for data in codec_data:
    #     if '11579_seg2' in data['snote_id_path']:
    #         score = pt.load_musicxml(data['score_path'], force_note_ids='keep')
    #         # score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    #         snote_ids = np.load(data['snote_id_path'])
    #         performed_part = pt.musicanalysis.decode_performance(score, parameters_to_performance_array(data['p_codec']), snote_ids=snote_ids)
    #         pt.save_performance_midi(performed_part, "tmp0.mid")
    #         hook()

    # score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml")
    # performance = pt.load_performance("../Datasets/vienna4x22/midi/Schubert_D783_no15_p01.mid")
    # _, alignment = pt.load_match("../Datasets/vienna4x22/match/Schubert_D783_no15_p01.match")
    # parameters, snote_ids, _ = pt.musicanalysis.encode_performance(score, performance, alignment)
    # performed_part = pt.musicanalysis.decode_performance(score, parameters, snote_ids=snote_ids)
    # pt.save_performance_midi(performed_part, "tmp0.mid")
    # score_part = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    # performed_part = render_sample(score_part, "logs/log_conv_transformer_melody_156/samples/samples_4000.npz.npy", "tmp.npy")
