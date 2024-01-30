"""Matrix (pianoroll converter) taken from SymRep project
Author: Huan Zhang
"""
import os
import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import rearrange, repeat
import pandas as pd

def load_data(path, cfg):
    """generic load data function for any type of representation """
    
    save_dir = cfg.experiment.data_save_dir
    if cfg.experiment.symrep in ["matrix", "sequence"]: # add further parameterized dirs for matrix and sequence
        save_dir = f"{save_dir}/{cfg[cfg.experiment.symrep].save_dir}"

    if not os.path.exists(save_dir):
        return None

    metadata = pd.read_csv(f"{save_dir}/metadata.csv")
    res = metadata[metadata['path'] == path]
    if len(res):
        if cfg.experiment.symrep == "graph":
            return np.array(dgl.load_graphs(f"{save_dir}/{res['save_dir'].iloc[0]}")[0])
        else:
            return np.load(f"{save_dir}/{res['save_dir'].iloc[0]}")


def save_data(path, computed_data, cfg):
    """generic save_data function for any type of representation
    - write the corresponding path with the saved index in metadata.csv
    
    graphs: dgl 
    matrix and sequence: numpy npy
    """

    save_dir = cfg.experiment.data_save_dir
    if cfg.experiment.symrep in ["matrix", "sequence"]: # add further parameterized dirs for matrix and sequence
        save_dir = f"{save_dir}/{cfg[cfg.experiment.symrep].save_dir}"

    if not os.path.exists(save_dir): # make saving dir if not exist
        os.makedirs(save_dir)
        with open(f"{save_dir}/metadata.csv", "w") as f:
            f.write("path,save_dir\n")

    metadata = pd.read_csv(f"{save_dir}/metadata.csv")
    if path in metadata['path']: # don't write and save if it existed
        return

    N = len(metadata) 
    if cfg.experiment.symrep == 'graph':
        save_path = f"{N}.dgl"
        dgl.save_graphs(f"{save_dir}/{save_path}", computed_data)
    else:
        save_path = f"{N}.npy"
        np.save(f"{save_dir}/{save_path}", computed_data)
    
    metadata = metadata.append({"path": path, "save_dir": save_path}, ignore_index=True)
    metadata.to_csv(f"{save_dir}/metadata.csv", index=False)



def midi_generate_rolls(note_events, pedal_events, cfg, duration=None):
    """Given the list of note_events, paint the rolls based on the duration of the segment and resolution
    Adapted from https://github.com/bytedance/piano_transcription/blob/master/utils/utilities.py
    """

    frames_num = cfg.matrix.resolution
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    velocity_roll = np.zeros((frames_num, cfg.matrix.bins))

    if not note_events:
        return onset_roll, velocity_roll
    
    start_delta = int(min([n.start for n in note_events]))
    if not duration:
        duration = note_events[-1].end - note_events[0].start + 1
    frames_per_second = (cfg.matrix.resolution / duration)

    for note_event in note_events:
        """note_event: e.g., Note(start=1.009115, end=1.066406, pitch=40, velocity=93)"""

        bgn_frame = min(int(round((note_event.start - start_delta) * frames_per_second)), frames_num-1)
        fin_frame = min(int(round((note_event.end - start_delta) * frames_per_second)), frames_num-1)
        velocity_roll[bgn_frame : fin_frame + 1, note_event.pitch] = (
            note_event.velocity if cfg.experiment.feat_level else 1)
        onset_roll[bgn_frame, note_event.pitch] = 1

    if cfg.experiment.feat_level:
        for pedal_event in pedal_events:
            """pedal_event: e.g., ControlChange(number=67, value=111, time=5.492188)"""

            if pedal_event.number == 64: ped_index = 128
            elif pedal_event.number == 66: ped_index = 129
            elif pedal_event.number == 67: ped_index = 130
            else: continue

            bgn_frame = min(int(round((pedal_event.time - start_delta) * frames_per_second)), frames_num-1)
            velocity_roll[bgn_frame : , ped_index] = pedal_event.value
            onset_roll[bgn_frame, ped_index] = 1

    return onset_roll, velocity_roll


def musicxml_generate_rolls(note_events, cfg):

    if len(note_events) == 0:
        return None

    start_delta = int(min([n['onset_div'] for n in note_events]))

    end_time_divs = note_events['onset_div'].max() + note_events['duration_div'].max()
    frames_num = cfg.matrix.resolution
    frames_per_second = (frames_num / end_time_divs)
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    voice_roll = np.zeros((frames_num, cfg.matrix.bins))

    for note_event in note_events:
        bgn_frame = min(int(round((note_event['onset_div'] - start_delta) * frames_per_second)), frames_num-1)
        fin_frame = min(bgn_frame + int(round((note_event['duration_div']) * frames_per_second)), frames_num-1)
        voice_roll[bgn_frame : fin_frame + 1, note_event['pitch']] = (
            note_event['voice'] if cfg.experiment.feat_level else 1)
        onset_roll[bgn_frame, note_event['pitch']] = 1

    # if cfg.experiment.feat_level:
    #     # add the score markings feature to matrix.
    #     raise NotImplementedError

    return onset_roll, voice_roll


def perfmidi_to_matrix(path, cfg):
    """Process MIDI events to roll matrices for training"""
    perf_data = pretty_midi.PrettyMIDI(path)

    note_events = perf_data.instruments[0].notes
    pedal_events = perf_data.instruments[0].control_changes

    if cfg.segmentation.seg_type == "fix_num":

        onset_roll, velocity_roll = midi_generate_rolls(note_events, pedal_events, cfg)
        onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
        velocity_roll = rearrange(velocity_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
    
    elif cfg.segmentation.seg_type == "fix_size":
        """in matrices, we define the <size> as amount of musical event"""
        onset_roll, velocity_roll = [], []
        __onset_append, __velocity_append = onset_roll.append, velocity_roll.append # this make things faster..
        """get segments by size and produce rolls"""
        for i in range(0, len(note_events), cfg.segmentation.seg_size):
            end = i+cfg.segmentation.seg_size
            seg_note_events = note_events[i:end]
            timings = [*map(lambda n: n.start, seg_note_events)]
            start, end = min(timings), max(timings)
            seg_pedal_events = [*filter(lambda p: (p.time > start and p.time < end)
                                , pedal_events)]
            seg_onset_roll, seg_velocity_roll = midi_generate_rolls(seg_note_events, seg_pedal_events, cfg)
            __onset_append(seg_onset_roll)
            __velocity_append(seg_velocity_roll)            
    
    elif cfg.segmentation.seg_type == "fix_time":  
        duration = cfg.segmentation.seg_time
        onset_roll, velocity_roll = [], []
        __onset_append, __velocity_append = onset_roll.append, velocity_roll.append # this make things faster..

        """get segment by time and produce rolls"""
        if cfg.segmentation.seg_hop:
            hop = cfg.segmentation.seg_hop
        else:
            hop = cfg.segmentation.seg_time
        for i in range(0, int(perf_data.get_end_time()), hop):
            start, end = i, i + cfg.segmentation.seg_time
            seg_note_events = [*filter(lambda n: (n.start > start and n.end < end) 
                                          , note_events)]  # losing the cross segment events..
            seg_pedal_events = [*filter(lambda p: p.time > start and p.time < end
                                           , pedal_events)]
            seg_onset_roll, seg_velocity_roll = midi_generate_rolls(seg_note_events, seg_pedal_events, cfg, duration=duration)
            __onset_append(seg_onset_roll)
            __velocity_append(seg_velocity_roll)

    matrices = torch.tensor(np.array([onset_roll, velocity_roll]))
    matrices = rearrange(matrices, "c s f n -> s c f n") # stack them in channel, c=2
    return matrices # (s 2 h w)


def musicxml_to_matrix(path, cfg):
    """Process musicXML to roll matrices for training"""

    import warnings
    warnings.filterwarnings("ignore") # mute partitura warnings

    try: # some parsing error....
        score_data = pt.load_musicxml(path)
        note_events = score_data.note_array()
    except Exception as e:
        print(f'failed on score {path} with exception {e}')
        return None

    if cfg.segmentation.seg_type == "fix_num":

        onset_roll, voice_roll = musicxml_generate_rolls(note_events, cfg)
        if 'asap-dataset/Schubert/Impromptu_op.90_D.899/4_' in path: # plotting
            from PIL import Image
            from matplotlib import cm
            # example = np.uint8(cm.rainbow(example)*255)
            im = Image.fromarray(np.uint8(cm.rainbow(example)*255))
        onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
        voice_roll = rearrange(voice_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)

        # im = Image.fromarray(np.uint8(cm.gist_earth(voice_roll)*255))
        # im.save('tmp.png')
    
    elif cfg.segmentation.seg_type == "fix_size":
        """in matrices, we define the <size> as amount of musical event"""
        onset_roll, voice_roll = [], []
        __onset_append, __voice_append = onset_roll.append, voice_roll.append # this make things faster..
        """get segments by size and produce rolls"""
        for i in range(0, len(note_events), cfg.segmentation.seg_size):
            end = i+cfg.segmentation.seg_size
            seg_note_events = note_events[i:end]
            res = musicxml_generate_rolls(seg_note_events, cfg)
            if res:
                seg_onset_roll, seg_voice_roll = res
                __onset_append(seg_onset_roll)
                __voice_append(seg_voice_roll)        
    
    elif cfg.segmentation.seg_type == "fix_time":  
        onset_roll, voice_roll = [], []
        __onset_append, __voice_append = onset_roll.append, voice_roll.append # this make things faster..
        """get segment by time (in beats) and produce rolls"""
        for i in range(0, int(max(note_events['onset_beat'])), cfg.segmentation.seg_beat):
            start, end = i, i + cfg.segmentation.seg_beat
            seg_note_events = note_events[(note_events['onset_beat'] > start) & (note_events['onset_beat'] < end)] # losing the cross segment events..
            res = musicxml_generate_rolls(seg_note_events, cfg)
            if res:
                seg_onset_roll, seg_voice_roll = res
                __onset_append(seg_onset_roll)
                __voice_append(seg_voice_roll)

    matrices = torch.tensor(np.array([onset_roll, voice_roll]))
    matrices = rearrange(matrices, "c s f n -> s c f n") # stack them in channel, c=2
    return matrices # (s 2 h w)

