import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

model_layers_and_heads_num_dict = {
    'llama3.1_8B_instruct': (32, 32),
    'llama3.1_70B_instruct': (80, 64),
    'internlm2_5-20b-chat': (48, 48),
    'Mistral-Nemo-Instruct-2407': (40, 32)
}

dataset_name_dict = {
    'belief': 'BE',
    'deception': 'DE',
    'harm': 'HA',
    'intention': 'IN',
    'known_knowns': 'KK',
    'known_unknowns': 'KU',
    'self_improve': 'SI',
    'self_reflection': 'SR',
    'sequential_planning': 'SP',
    'situational_awareness': 'SA'
}

task_rung_mapping = {
    "c1": ["SA", "SP", "BE", "IN"],
    "c2": ["SR", "SI", "DE", "SP", "KK", "KU", "HA"],
}

def get_label_color(label, scenario_ladder_mapping):
    colors = {"c1": "#a45077", "c2": "#4678b0"}
    for key, values in scenario_ladder_mapping.items():
        if label in values:
            return colors[key]
    return "black"

c1_datasets = ['situational_awareness', 'sequential_planning', 'belief', 'intention']

c2_datasets = [dataset for dataset in dataset_name_dict if dataset not in c1_datasets]

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
args = parser.parse_args()

main_path = os.path.join(os.path.dirname(__file__), 'exp_each_head')
paths = os.listdir(main_path)

vmin = 0.5
vmax = 0.8

alpha = 1
linewidth = 1.2

title_size = 20

num_x_ticks = model_layers_and_heads_num_dict[args.model_name][1] // 4 + 1
num_y_ticks = model_layers_and_heads_num_dict[args.model_name][0] // 4 + 1

edgecolor1 = 'red'
edgecolor2 = 'blue'

fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1.25])
axes = []

c1_paths = []
c2_paths = []
for path in paths:
    if path.startswith(args.model_name):
        exp_name = path[len(args.model_name) + 1:]
        if exp_name in c1_datasets:
            c1_paths.append((exp_name, path))
        elif exp_name in c2_datasets:
            c2_paths.append((exp_name, path))

c1_paths.sort()
c2_paths.sort()

for i, (exp_name, path) in enumerate(c1_paths[:4]):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    axes.append(ax)
    now_path = os.path.join(main_path, path, path + '_head_wise_accuracies.json')

    with open(now_path, 'r') as json_file:
        data = json.load(json_file)

    data_array = np.array(data).reshape(model_layers_and_heads_num_dict[args.model_name])

    if (i + 1) % 2 == 0:
        sns.heatmap(data_array, ax=ax, cmap='Greens', vmin=vmin, vmax=vmax, cbar=False)
    else:
        sns.heatmap(data_array, ax=ax, cmap='Greens', vmin=vmin, vmax=vmax,
                    cbar=False)

    ax.set_title(exp_name.replace('_', ' ').title(), fontsize=title_size, color=get_label_color(dataset_name_dict[exp_name],task_rung_mapping), fontweight='bold')
    x_ticks = np.linspace(0, data_array.shape[1], num_x_ticks, dtype=int)
    y_ticks = np.linspace(0, data_array.shape[0], num_y_ticks, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax.set_xticklabels(x_ticks, rotation=0)
    ax.set_yticklabels(y_ticks, rotation=0)

    max_vals = np.argsort(-data_array, axis=None)[:100]
    min_vals = np.argsort(data_array, axis=None)[:100]

    for val in max_vals:
        row, col = np.unravel_index(val, data_array.shape)
        rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor=edgecolor1, linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)

    for val in min_vals:
        row, col = np.unravel_index(val, data_array.shape)
        rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor=edgecolor2, linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)

for i, (exp_name, path) in enumerate(c2_paths[:6]):
    ax = fig.add_subplot(gs[i // 3, i % 3 + 2])
    axes.append(ax)
    now_path = os.path.join(main_path, path, path + '_head_wise_accuracies.json')

    with open(now_path, 'r') as json_file:
        data = json.load(json_file)

    data_array = np.array(data).reshape(model_layers_and_heads_num_dict[args.model_name])

    if (i + 1) % 3 == 0:
        sns.heatmap(data_array, ax=ax, cmap='Greens', vmin=vmin, vmax=vmax, cbar=True)
    else:
        sns.heatmap(data_array, ax=ax, cmap='Greens', vmin=vmin, vmax=vmax,
                    cbar=False)

    ax.set_title(exp_name.replace('_', ' ').title(), fontsize=title_size, color=get_label_color(dataset_name_dict[exp_name],task_rung_mapping), fontweight='bold')
    x_ticks = np.linspace(0, data_array.shape[1], num_x_ticks, dtype=int)
    y_ticks = np.linspace(0, data_array.shape[0], num_y_ticks, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax.set_xticklabels(x_ticks, rotation=0)
    ax.set_yticklabels(y_ticks, rotation=0)

    max_vals = np.argsort(-data_array, axis=None)[:100]
    min_vals = np.argsort(data_array, axis=None)[:100]

    for val in max_vals:
        row, col = np.unravel_index(val, data_array.shape)
        rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor=edgecolor1, linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)

    for val in min_vals:
        row, col = np.unravel_index(val, data_array.shape)
        rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor=edgecolor2, linewidth=linewidth, alpha=alpha)
        ax.add_patch(rect)

fig.text(0.5, 0.01, 'Head Index', ha='center', fontsize=20)
fig.text(0.01, 0.5, 'Layer Index', va='center', rotation='vertical', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(left=0.042, bottom=0.07)

save_path = os.path.join(os.path.dirname(__file__), 'pic_heatmap', args.model_name)
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, args.model_name + '_heatmaps.jpg'), dpi=300)
plt.savefig(os.path.join(save_path, args.model_name + '_heatmaps.pdf'), format='pdf')
plt.show()
