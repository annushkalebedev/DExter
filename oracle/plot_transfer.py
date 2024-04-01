import os, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import hook

DIR = "/homes/hz009/Research/DExter/artifacts/samples/EVAL-targetgen_noise-lw11111-len200-beta0.02-steps1000-epsilon-TransferTrue-ssfrac0.25-cfdg_ddpm-w=1.2-dim=48"

all_paths = glob.glob(f"{DIR}/**/*midlevel.csv", recursive=True)
label_paths = sorted([p for p in all_paths if 'label' in p])
source_paths = sorted([p for p in all_paths if 'source' in p])
pred_paths =  sorted([p for p in all_paths if (not 'label' in p) and (not 'source' in p)])

hook()
label_feats = pd.concat([pd.read_csv(p, index_col=0) for p in label_paths])
source_feats = pd.concat([pd.read_csv(p, index_col=0) for p in source_paths])
pred_feats = pd.concat([pd.read_csv(p, index_col=0) for p in pred_paths])

label_feats['type'] = 'label'
source_feats['type'] = 'source'
pred_feats['type'] = 'pred'

df = pd.concat([label_feats, source_feats, pred_feats])
df = df.melt(id_vars=['type'], value_vars=['melodiousness', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness'])


# Calculate the mean values for each type and variable
mean_values = df.groupby(['type', 'variable']).value.mean().unstack(0)

# Calculate the differences
mean_values['pred_label_diff'] = mean_values['pred'] - mean_values['label']
mean_values['pred_source_diff'] = mean_values['pred'] - mean_values['source']

# Prepare data for the radar chart
features = mean_values.index
num_features = len(features)

# Angle of each axis in the plot (divide the full circle into equal parts)
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Radar chart data
data_pred_label = mean_values['pred_label_diff'].tolist()
data_pred_source = mean_values['pred_source_diff'].tolist()
data_pred_label += data_pred_label[:1]  # Complete the loop
data_pred_source += data_pred_source[:1]  # Complete the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one line per variable + fill
ax.plot(angles, data_pred_label, color='red', linewidth=2, label='Pred - Label')
ax.fill(angles, data_pred_label, color='red', alpha=0.25)
ax.plot(angles, data_pred_source, color='blue', linewidth=2, label='Pred - Source')
ax.fill(angles, data_pred_source, color='blue', alpha=0.25)

# Add feature names as labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)

# Add a title, legend, and make sure the plot is circular by setting the aspect ratio
ax.set_title('Feature Difference Radar Chart', size=15, color='black', position=(0.5, 1.1))
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.show()



hook()