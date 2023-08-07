import os, sys, glob
import warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt

import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from tqdm import tqdm
import hook


VIENNA_MATCH_DIR = "../Datasets/vienna4x22/match/"
VIENNA_MUSICXML_DIR = "../Datasets/vienna4x22/musicxml/"
VIENNA_PERFORMANCE_DIR = "../Datasets/vienna4x22/midi/"

ASAP_DIR = "../Datasets/asap-dataset-alignment/"
ASAP_MATCH = "../Datasets/asap-dataset-alignment/Scriabin/Sonatas/5/Yeletskiy07M_note_alignments/note_alignment.tsv"
ASAP_MUSICXML = "../Datasets/asap-dataset-alignment/Scriabin/Sonatas/5/xml_score.musicxml"
ASAP_PERFORMANCE = "../Datasets/asap-dataset-alignment/Scriabin/Sonatas/5/Yeletskiy07M.mid"

BMZ_MATCH_DIR = "../Datasets/pianodata-master/match"
BMZ_MUSICXML_DIR = "../Datasets/pianodata-master/xml/"

ATEPP_DIR = "../Datasets/ATEPP-1.1"
ATEPP_META_DIR = "../Datasets/ATEPP-1.1/ATEPP-metadata-1.3.csv"

MAX_NOTE_LEN = 1000


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


def process_dataset_codec():
    """process the performance features for the given dataset. Save the 
    computed features in the form of numpy arrays in the same directory as 
    performance data.

    Args:
        dataset (str): dataset to process. Defaults to 'ASAP'.
    """

    prev_s_path, data = None, []
    for dataset in ['ATEPP', 'ASAP', 'VIENNA422']:

        if dataset == "VIENNA422":
            alignment_paths = glob.glob(os.path.join(VIENNA_MATCH_DIR, "*[!e].match"))
            score_paths = [(VIENNA_MUSICXML_DIR + pp.split("/")[-1][:-10] + ".musicxml") for pp in alignment_paths]
            performance_paths = [None] * len(alignment_paths) # don't use the given performance, use the aligned.
        if dataset == "ASAP":
            performance_paths = glob.glob(os.path.join(ASAP_DIR, "**/*[!e].mid"), recursive=True)
            alignment_paths = [(pp[:-4] + "_note_alignments/note_alignment.tsv") for pp in performance_paths]
            score_paths = [os.path.join("/".join(pp.split("/")[:-1]), "xml_score.musicxml") for pp in performance_paths]
        if dataset == "ATEPP":
            alignment_paths = glob.glob(os.path.join(ATEPP_DIR, "**/[!z]*n.csv"), recursive=True)
            performance_paths = [(aa[:-10] + ".mid") for aa in alignment_paths]
            score_paths = [glob.glob(os.path.join("/".join(pp.split("/")[:-1]), "*.*l"))[0] for idx, pp in enumerate(performance_paths)]
        # if dataset == "BMZ":
        #     alignment_paths = glob.glob(os.path.join(BMZ_MATCH_DIR, "**/*.match"), recursive=True)
        #     alignment_paths = [p for p in alignment_paths if (("Take" not in p) and ("mozart" not in p))]
        #     score_paths = [BMZ_MUSICXML_DIR + ap.split("/")[-1][:-6] + ".xml" for ap in alignment_paths]
        #     performance_paths = [None] * len(alignment_paths) # don't use the given performance, use the aligned.
            
        for s_path, p_path, a_path in tqdm(zip(score_paths, performance_paths, alignment_paths)):

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
                save_snote_id_path = f"data/snote_ids/N={MAX_NOTE_LEN}/{dataset}_{piece_name}"

                # encode!
                if prev_s_path == s_path:
                    p_codec, score, snote_ids = get_performance_codec(s_path, a_path, p_path, score=score)
                else:
                    p_codec, score, snote_ids = get_performance_codec(s_path, a_path, p_path)
                # snote_ids = list(set(snote_ids))
                p_codec = rfn.structured_to_unstructured(p_codec)

                sna = score.note_array()
                sna = sna[np.in1d(sna['id'], snote_ids)]
                s_codec = rfn.structured_to_unstructured(
                    sna[['onset_div', 'duration_div', 'pitch', 'voice']])

                if not ((p_codec.shape == s_codec.shape) and (len(p_codec) == len(snote_ids))):
                    print(f"{a_path} has length issue: p: {len(p_codec)}; s: {len(s_codec)}") 
                    continue

                # split the piece 
                for idx in range(0, len(p_codec), MAX_NOTE_LEN):
                    seg_p_codec = p_codec[idx : idx + MAX_NOTE_LEN]
                    seg_s_codec = s_codec[idx : idx + MAX_NOTE_LEN]
                    seg_snote_ids = snote_ids[idx : idx + MAX_NOTE_LEN]

                    if len(seg_p_codec) < MAX_NOTE_LEN:
                        seg_p_codec = np.pad(seg_p_codec, ((0, MAX_NOTE_LEN - len(seg_p_codec)), (0, 0)), mode='constant', constant_values=0)
                        seg_s_codec = np.pad(seg_s_codec, ((0, MAX_NOTE_LEN - len(seg_s_codec)), (0, 0)), mode='constant', constant_values=0)

                    if len(seg_snote_ids) == 0:
                        hook()

                    seg_id_path = f"{save_snote_id_path}_seg{int(idx/MAX_NOTE_LEN)}.npy"
                    # save snote_id
                    np.save(seg_id_path, seg_snote_ids) 

                    data.append({"p_codec": seg_p_codec, 
                                "s_codec": seg_s_codec,
                                "snote_id_path": seg_id_path,
                                "score_path": s_path,
                                "piece_name": piece_name  # piece name for shortcut and identifying the generated sample
                                })

                prev_s_path = s_path
            else:
                print(f"Data incomplete for {a_path}")


    # print(max([data[i].shape[0] for i in range(22)]))
    np.save(f"data/codec_N={MAX_NOTE_LEN}.npy", np.stack(data))

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
    parameters, snote_ids, pad_mask = pt.musicanalysis.encode_performance(score, performance, alignment)

    return parameters, score, snote_ids


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
    codec_data = np.load(f"./data/p_codec_N=100.npy", allow_pickle=True) # (N_data, 1000, 4)
    p_codecs = [cd['p_codec'] for cd in codec_data]



def plot_codec(data, ax0, ax1, fig):
    # plot the p_codec and s_codec, on the given two axes

    p_im = ax0.imshow(data["p_codec"].T, aspect='auto', origin='lower')
    ax0.set_yticks([0, 1, 2, 3])
    ax0.set_yticklabels(["beat_period", "velocity", "timing", "articulation_log"])
    fig.colorbar(p_im, orientation='vertical', ax=ax0)
    ax0.set_title(data['snote_id_path'])

    s_im = ax1.imshow(data['s_codec'].T, aspect='auto', origin='lower')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['onset_div', 'duration_div', 'pitch', 'voice'])
    fig.colorbar(s_im, orientation='vertical', ax=ax1)
    
    return 


def plot_codec_list(codec_list):

    n_data = len(codec_list)
    fig, ax = plt.subplots(2 * n_data, 1, figsize=(24, 4 * n_data))
    for idx, data in enumerate(codec_list):
        plot_codec(data, ax[idx * 2], ax[idx * 2 + 1], fig)

    plt.savefig("tmp.png")

    return fig


if __name__ == '__main__':


    # process_dataset_codec()
    # codec_data_analysis()

    # from utils import parameters_to_performance_array
    codec_data = np.load("data/codec_N=100.npy", allow_pickle=True) 

    plot_codec_list(codec_data[-110:-105])

    # score = pt.load_musicxml(data['score_path'], force_note_ids='keep')
    # # score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    # snote_ids = np.load(data['snote_id_path'])
    # N = len(snote_ids)
    # performed_part = pt.musicanalysis.decode_performance(score, parameters_to_performance_array(data['p_codec'])[:N], snote_ids=snote_ids)
    # pt.save_performance_midi(performed_part, "tmp0.mid")
    hook()

    # score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml", force_note_ids=True)
    # score_part = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    # performed_part = render_sample(score_part, "logs/log_conv_transformer_melody_156/samples/samples_4000.npz.npy", "tmp.npy")
