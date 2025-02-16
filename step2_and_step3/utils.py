import numpy as np
import torch
from baukit import TraceDict
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def format_get_activation_dataset(question, choice):
    return f"Answer the following question with only the most correct option and no extra content.\n{question}\nAnswer: {choice}"


def format_benchmark_dataset(question):
    return f"Answer the following question with only the most correct option and no extra content.\n{question}\nAnswer: "


def tokenized_dataset(dataset, tokenizer, benchmark=False):
    all_prompts = []
    all_labels = []
    if benchmark:
        for i in range(len(dataset)):
            question = dataset[i]['question']
            prompt = format_benchmark_dataset(question)
            if i == 0:
                print(prompt)
            prompt = tokenizer(prompt, return_tensors='pt').input_ids
            all_prompts.append(prompt)
        return all_prompts, all_labels
    else:
        for i in range(len(dataset)):
            question = dataset[i]['question']
            answer_type = ['false_answer', 'true_answer']
            for j in range(2):
                prompt = format_get_activation_dataset(question, dataset[i][answer_type[j]])
                if i == 0 and j == 0:
                    print(prompt)
                prompt = tokenizer(prompt, return_tensors='pt').input_ids
                all_prompts.append(prompt)
                all_labels.append(j)

        return all_prompts, all_labels


def get_llama_activations_bau(model, prompt, device):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS + MLPS) as ret:
            output = model(prompt, output_hidden_states=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def get_internlm_activations_bau(model, prompt, device):
    HEADS = [f"model.layers.{i}.attention.wo" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.feed_forward.w2" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS + MLPS) as ret:
            output = model(prompt, output_hidden_states=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def get_mistral_activations_bau(model, prompt, device):
    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS + MLPS) as ret:
            output = model(prompt, output_hidden_states=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir,
                           com_directions):
    interventions = {}
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        if use_center_of_mass:
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir:
            direction = np.random.normal(size=(128,))
        else:
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:, layer, head, :]  # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze(), proj_val_std))

    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0])
    return interventions

def get_intern_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir,
                           com_directions):
    interventions = {}
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.attention.wo"] = []
    for layer, head in top_heads:
        if use_center_of_mass:
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir:
            direction = np.random.normal(size=(128,))
        else:
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:, layer, head, :]  # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.attention.wo"].append((head, direction.squeeze(), proj_val_std))

    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.attention.wo"] = sorted(
            interventions[f"model.layers.{layer}.attention.wo"], key=lambda x: x[0])
    return interventions


def train_probes(seed, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers,
                 num_heads):
    all_head_accs = []
    probes = []

    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis=0)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis=0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis=0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis=0)

    for layer in tqdm(range(num_layers), desc="train_probes"):
        for head in range(num_heads):
            X_train = all_X_train[:, layer, head, :]
            X_val = all_X_val[:, layer, head, :]

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def get_top_heads(train_idxs, val_idxs, separated_activations, separated_labels, num_layers, num_heads, seed,
                  num_to_intervene, use_random_dir=False):
    probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels,
                                            num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir:
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads * num_layers, num_heads * num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, probes


def get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations,
                       separated_labels):
    com_directions = []

    for layer in tqdm(range(num_layers), desc="get_com_directions"):
        for head in range(num_heads):
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = np.concatenate(
                [separated_head_wise_activations[i][:, layer, head, :] for i in usable_idxs], axis=0)
            usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions
