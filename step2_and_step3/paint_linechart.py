import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

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
c1_datasets.sort()

c2_datasets = [dataset for dataset in dataset_name_dict if dataset not in c1_datasets]
c2_datasets.sort()

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, nargs='+', default=[
    'my_llama3.1_8B_instruct',
    'my_llama3.1_70B_instruct',
    'internlm2_5-20b-chat',
    'Mistral-Nemo-Instruct-2407'
])

formal_name_dict = {
    'my_llama3.1_8B_instruct': 'Llama3.1-8B-Instruct',
    'my_llama3.1_70B_instruct': 'Llama3.1-70B-Instruct',
    'internlm2_5-20b-chat': 'InternLM2.5-20B-Chat',
    'Mistral-Nemo-Instruct-2407': 'Mistral-Nemo-Instruct'
}

args = parser.parse_args()

main_path = os.path.join(os.path.dirname(__file__), 'exp_each_head')
paths = os.listdir(main_path)

colors = sns.color_palette("husl", len(args.models))

fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharey=False)

filtered_paths = {model: [] for model in args.models}
for key in dataset_name_dict.keys():
    for path in paths:
        for model_name in args.models:
            if path.startswith(model_name) and key in path:
                exp_name = path[len(model_name) + 1:]
                filtered_paths[model_name].append((exp_name, path))
                break
for i, dataset in enumerate(c1_datasets):
    ax = axes[i // 2, i % 2]

    for model_idx, model_name in enumerate(args.models):
        exp_name, path = next((name, p) for name, p in filtered_paths[model_name] if dataset in name)
        now_path = os.path.join(main_path, path, path + '_head_wise_accuracies.json')

        with open(now_path, 'r') as json_file:
            data = json.load(json_file)

        layers, heads = model_layers_and_heads_num_dict[model_name]
        data_array = np.array(data).reshape((layers, heads))

        layer_means = data_array.mean(axis=1)

        smoothed_layer_means = gaussian_filter1d(layer_means, sigma=1)

        layer_std = data_array.std(axis=1)
        lower_bound = smoothed_layer_means - layer_std
        upper_bound = smoothed_layer_means + layer_std

        x = np.arange(1, layers + 1) / layers

        ax.plot(x, smoothed_layer_means, label=model_name, color=colors[model_idx], linewidth=2)

        ax.fill_between(x, lower_bound, upper_bound, color=colors[model_idx], alpha=0.2)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(dataset.replace('_', ' ').title(), fontsize=17, color=get_label_color(dataset_name_dict[dataset],task_rung_mapping), fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)

for i, dataset in enumerate(c2_datasets):
    ax = axes[i // 3, i % 3 + 2]

    for model_idx, model_name in enumerate(args.models):
        exp_name, path = next((name, p) for name, p in filtered_paths[model_name] if dataset in name)
        now_path = os.path.join(main_path, path, path + '_head_wise_accuracies.json')

        with open(now_path, 'r') as json_file:
            data = json.load(json_file)

        layers, heads = model_layers_and_heads_num_dict[model_name]
        data_array = np.array(data).reshape((layers, heads))

        layer_means = data_array.mean(axis=1)

        smoothed_layer_means = gaussian_filter1d(layer_means, sigma=1)

        layer_std = data_array.std(axis=1)
        lower_bound = smoothed_layer_means - layer_std
        upper_bound = smoothed_layer_means + layer_std

        x = np.arange(1, layers + 1) / layers

        ax.plot(x, smoothed_layer_means, label=model_name, color=colors[model_idx], linewidth=2)

        ax.fill_between(x, lower_bound, upper_bound, color=colors[model_idx], alpha=0.2)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(dataset.replace('_', ' ').title(), fontsize=17, color=get_label_color(dataset_name_dict[dataset],task_rung_mapping), fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)

handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, [formal_name_dict[x] for x in labels], loc='lower center', ncol=len(args.models),
           fontsize=18, frameon=False)

fig.text(0.5, 0.09, 'Relative Layer Index', ha='center', fontsize=18)
fig.text(0.01, 0.5, 'Mean Accuracy', va='center', rotation='vertical', fontsize=18)
plt.tight_layout(rect=[0.02, 0.1, 1, 1])

save_path = os.path.join(os.path.dirname(__file__), 'pic_linechart', '_'.join(args.models))
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, 'comparison_line_plots.png'))
plt.savefig(os.path.join(save_path, 'comparison_line_plots.pdf'), format='pdf')
plt.show()
