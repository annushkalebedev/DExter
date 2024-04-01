import os, glob
import torch
import pandas as pd
from omegaconf import OmegaConf
from matrix import perfmidi_to_matrix
from train_oracle import MIDItoCEPModel
from tqdm import tqdm
import hook

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

def predict_from_midi(midi_path, model_path, model):

    # Convert MIDI to pianoroll matrix
    pianoroll = perfmidi_to_matrix(midi_path, convert_cfg)

    # Convert to PyTorch tensor and add batch dimension
    pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        prediction = model(pianoroll_tensor)
        prediction = prediction.detach().numpy()

    return prediction


# DIR = "/homes/hz009/Research/DExter/artifacts/samples/EVALo-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferFalse-ssfrac1-cfdg_ddpm-w=1.2-dim=48/epoch=0"
DIR = "/homes/hz009/Research/DExter/artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac1-cfdg_ddpm-w=1.2-dim=48"
model_path = '/homes/hz009/Research/DExter/oracle/checkpoints/best-checkpoint.ckpt'

# Load the trained model
model = MIDItoCEPModel.load_from_checkpoint(model_path)
model.eval()

for midi_path in tqdm(glob.glob(f"{DIR}/**/*.mid", recursive=True)):
    if os.path.exists(midi_path[:-4] + "_midlevel.csv"):
        continue

    try:
        prediction = predict_from_midi(midi_path, model_path, model)
        prediction = pd.DataFrame(prediction, columns=['melodiousness', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness'])
        prediction.to_csv(midi_path[:-4] + "_midlevel.csv")
    except Exception as e:
        print(e)
