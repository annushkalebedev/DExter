import argparse
import os, sys, glob
from multiprocessing import Pool
sys.path.insert(0, "../partitura")
sys.path.insert(0, "../")
import partitura as pt

import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
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


def process_dataset_codec(dataset='ASAP'):
    """process the performance features for the given dataset. Save the 
    computed features in the form of numpy arrays in the same directory as 
    performance data.

    Args:
        dataset (str): dataset to process. Defaults to 'ASAP'.
    """

    if dataset == "VIENNA422":
        performance_paths = glob.glob(os.path.join(VIENNA_PERFORMANCE_DIR, "*[!e].mid"))
        # performance_paths = [pp for pp in performance_paths if "Schubert" in pp] # try only schubert data in the beginning
        alignment_paths = [(VIENNA_MATCH_DIR + pp.split("/")[-1][:-4] + ".match") for pp in performance_paths]
        score_paths = [(VIENNA_MUSICXML_DIR + pp.split("/")[-1][:-8] + ".musicxml") for pp in performance_paths]
        performance_paths = [None] * len(alignment_paths) # don't use the given performance, use the aligned.
    if dataset == "ASAP":
        performance_paths = glob.glob(os.path.join(ASAP_DIR, "**/*[!e].mid"), recursive=True)[1040:]
        alignment_paths = [(pp[:-4] + "_note_alignments/note_alignment.tsv") for pp in performance_paths]
        score_paths = [os.path.join("/".join(pp.split("/")[:-1]), "xml_score.musicxml") for pp in performance_paths]
    if dataset == "ATEPP":
        alignment_paths = glob.glob(os.path.join(ATEPP_DIR, "**/*_match.txt"), recursive=True)
        performance_paths = [(aa[:-10] + ".mid") for aa in alignment_paths]
        score_paths = [glob.glob(os.path.join("/".join(pp.split("/")[:-1]), "*xml"))[0] for pp in performance_paths]
    if dataset == "BMZ":
        alignment_paths = glob.glob(os.path.join(BMZ_MATCH_DIR, "**/*.match"), recursive=True)
        alignment_paths = [p for p in alignment_paths if (("Take" not in p) and ("mozart" not in p))]
        score_paths = [BMZ_MUSICXML_DIR + ap.split("/")[-1][:-6] + ".xml" for ap in alignment_paths]
        performance_paths = [None] * len(alignment_paths) # don't use the given performance, use the aligned.
    

    prev_s_path, data = None, []
    for s_path, p_path, a_path in tqdm(zip(score_paths, performance_paths, alignment_paths)):

        # parsing error
        if s_path == '../Datasets/ATEPP-1.1/Frederic_Chopin/Scherzo_No._4_in_E_Major,_Op._54,_B._148/score.xml':
            continue
        if s_path == '../Datasets/ATEPP-1.1/Frederic_Chopin/Nocturne_No.13_in_C_minor,_Op._48_No._1/score.xml':
            continue
        if s_path == '../Datasets/ATEPP-1.1/Frederic_Chopin/24_Preludes,_Op._28/No._7_in_A_Major:_Andantino/score.xml':
            continue
        if s_path == '../Datasets/ATEPP-1.1/Wolfgang_Amadeus_Mozart/Piano_Sonata_No.4_in_E_flat,_K.282/2._Menuetto_I-II/score.xml':
            continue
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

        if (os.path.exists(s_path) and os.path.exists(a_path)):

            if prev_s_path == s_path:
                p_codec, score, save_snote_id_path = get_performance_codec(s_path, a_path, p_path, score=score)
            else:
                p_codec, score, save_snote_id_path = get_performance_codec(s_path, a_path, p_path)

            p_codec = rfn.structured_to_unstructured(p_codec)
            s_codec = rfn.structured_to_unstructured(
                score.note_array()[['onset_div', 'duration_div', 'pitch', 'voice']])
            
            p_codec = np.pad(p_codec, ((0, 1000-len(p_codec)), (0, 0)), mode='constant', constant_values=0)
            s_codec = np.pad(s_codec, ((0, 1000-len(s_codec)), (0, 0)), mode='constant', constant_values=0)
            assert (p_codec.shape == s_codec.shape)

            if dataset == "VIENNA422":
                piece_name = s_path.split("/")[-1].split(".")[0]
            data.append({"p_codec": p_codec, 
                         "s_codec": s_codec,
                         "snote_id_path": save_snote_id_path,
                         "score_path": s_path,
                         "piece_name": piece_name  # piece name for shortcut and identifying the generated sample
                         })

            if np.isnan(p_codec).any():
                hook()
            prev_s_path = s_path
        else:
            print(f"Data incomplete for {a_path}")


    # print(max([data[i].shape[0] for i in range(22)]))
    np.save("data/vienna422_codec.npy", np.stack(data))
    return 


def get_performance_codec(score_path, alignment_path, performance_path=None, score=None):
    """compute the performance feature given score, alignment and performance path.
    Args:
        dataset (str, optional): _description_. Defaults to 'ASAP'.
    """

    if isinstance(score, type(None)):
        score = pt.load_musicxml(score_path)

    if alignment_path[-5:] == "match":
        performance, alignment = pt.load_match(alignment_path)
    elif alignment_path[-3:] == "tsv":
        alignment = pt.io.importparangonada.load_alignment_from_ASAP(alignment_path)

    if not isinstance(performance_path, type(None)): # use the performance if it's given
        performance = pt.load_performance(performance_path)

    # get the performance encodings
    parameters, snote_ids, pad_mask = pt.musicanalysis.encode_performance(score, performance, alignment)
    
    # save snote_id if it doesn't exist already
    save_snote_id_path = score_path.split("/")[-1].split(".")[0]
    save_snote_id_path = f"data/snote_ids/vienna422_{save_snote_id_path}.npy"
    if not os.path.exists(save_snote_id_path):
        np.save(save_snote_id_path, snote_ids)

    return parameters, score, save_snote_id_path


def render_sample(score_part, sample_path, snote_ids_path):
    """render """
    snote_ids = np.load(snote_ids_path)
    for idx in range(32):
        performance_array = reverse_quantized_codec(np.load(sample_path)[idx])
        performed_part = pt.musicanalysis.decode_performance(score_part, performance_array, snote_ids=snote_ids)

        pt.save_performance_midi(performed_part, f"samples/sample_{idx}.mid")

    return performed_part


if __name__ == '__main__':


    # score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml")

    # performance = pt.load_performance("../Datasets/vienna4x22/midi/Schubert_D783_no15_p01.mid")
    # _, alignment = pt.load_match("../Datasets/vienna4x22/match/Schubert_D783_no15_p01.match")
    # parameters, snote_ids, pad_mask = pt.musicanalysis.encode_performance(score, performance, alignment)
    # performed_part = pt.musicanalysis.decode_performance(score, parameters, snote_ids=snote_ids, pad_mask=pad_mask)
    # pt.save_performance_midi(performed_part, "tmp.mid")

    process_dataset_codec(dataset="VIENNA422")

    # score = pt.load_musicxml("../Datasets/vienna4x22/musicxml/Schubert_D783_no15.musicxml", force_note_ids=True)
    # score_part = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)) 
    # performed_part = render_sample(score_part, "logs/log_conv_transformer_melody_156/samples/samples_4000.npz.npy", "tmp.npy")
