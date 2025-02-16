import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
from baukit import TraceDict
from einops import rearrange
from datasets import load_dataset
from pandas import DataFrame
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_interventions_dict, get_top_heads, get_com_directions, tokenized_dataset, \
    get_intern_interventions_dict

base_path = os.path.dirname(__file__)

HF_NAMES = {
    'llama3.1_8B_instruct': 'path/to/original-hf-model/Meta-Llama-3.1-8B-Instruct',
    'llama3.1_70B_instruct': 'path/to/original-hf-model/Llama-3.1-8B-Instruct',
    'internlm2_5-20b-chat': 'path/to/original-hf-model/internlm2_5-20b-chat',
    'Mistral-Nemo-Instruct-2407': 'path/to/original-hf-model/Mistral-Nemo-Instruct-2407'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct', choices=HF_NAMES.keys(),
                        help='model name')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size',
                        default=0.2)
    parser.add_argument('--use_center_of_mass', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')

    args = parser.parse_args()
    device = 'cuda'
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_name = 'data'

    model_name_or_path = HF_NAMES[args.model_name]
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    for dataset_name in os.listdir(os.path.join(base_path, data_name)):
        for alpha in [0, 5, 10, 15, 20, 25, 30, 35]:
            args.alpha = alpha
            args.dataset_name = dataset_name[:-5]
            dataset = load_dataset('json', data_files=os.path.join(base_path, data_name, args.dataset_name + '.json'))[
                'train']

            save_path = os.path.join(base_path,
                                     f'step3_exp',
                                     f'step3{args.model_name[:6]}_{args.num_heads}heads_{args.use_center_of_mass}_mass',
                                     f'alpha_{args.alpha}', args.dataset_name)
            os.makedirs(save_path, exist_ok=True)
            json_save_path = os.path.join(save_path, args.model_name + '.json')
            if os.path.exists(json_save_path):
                with open(json_save_path, 'r') as file:
                    file_lines = sum(1 for line in file)
                if file_lines == len(dataset):
                    print(args.dataset_name + ' has completed.')
                    continue
                else:
                    os.remove(json_save_path)

            fold_idxs = np.array_split(np.arange(len(dataset)), args.num_fold)

            # load activations
            head_wise_activations = np.load(
                os.path.join(base_path, 'features', f'{args.model_name}_{args.dataset_name}_head_wise.npy'))
            labels = np.load(os.path.join(base_path, 'features', f'{args.model_name}_{args.dataset_name}_labels.npy'))
            head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

            activations_dataset = args.dataset_name
            tuning_activations = head_wise_activations

            separated_head_wise_activations = rearrange(head_wise_activations, '(b c) l h d -> b c l h d', c=2)
            separated_labels = [[0, 1]] * len(dataset)

            # run k-fold cross validation
            results = []
            for i in range(args.num_fold):
                train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
                test_idxs = fold_idxs[i]

                print(f"Running fold {i}")

                # pick a val set using numpy
                train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * (1 - args.val_ratio)),
                                                  replace=False)
                val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

                # save train and test splits
                df = DataFrame(dataset)
                splits_path = os.path.join(base_path, 'splits')
                os.makedirs(splits_path, exist_ok=True)
                df.iloc[train_set_idxs].to_csv(f"{splits_path}/fold_{i}_train_seed_{args.seed}.csv", index=False)
                df.iloc[val_set_idxs].to_csv(f"{splits_path}/fold_{i}_val_seed_{args.seed}.csv", index=False)
                df.iloc[test_idxs].to_csv(f"{splits_path}/fold_{i}_test_seed_{args.seed}.csv", index=False)

                # get directions
                if args.use_center_of_mass:
                    com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs,
                                                        separated_head_wise_activations, separated_labels)
                else:
                    com_directions = None
                top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations,
                                                  separated_labels, num_layers, num_heads, args.seed, args.num_heads,
                                                  args.use_random_dir)

                print("Heads intervened: ", sorted(top_heads))

                if 'intern' in args.model_name:
                    interventions = get_intern_interventions_dict(top_heads, probes, tuning_activations, num_heads,
                                                                  args.use_center_of_mass, args.use_random_dir,
                                                                  com_directions)
                else:
                    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads,
                                                           args.use_center_of_mass, args.use_random_dir, com_directions)

                def lt_modulated_vector_add(head_output, layer_name):
                    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                    for head, direction, proj_val_std in interventions[layer_name]:
                        direction_to_add = torch.tensor(direction).to(head_output.device.index)
                        head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                    return head_output

                filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'

                if args.use_center_of_mass:
                    filename += '_com'
                if args.use_random_dir:
                    filename += '_random'

                prompts, _ = tokenized_dataset(dataset, tokenizer, benchmark=True)
                with open(json_save_path, "a") as file:
                    for j in tqdm(test_idxs.tolist()):
                        prompt = prompts[j].to(device)
                        with TraceDict(model, list(interventions.keys()), edit_output=lt_modulated_vector_add) as ret:
                            model_gen_tokens = model.generate(prompt, top_k=1, max_length=prompt.shape[-1] + 3,
                                                              num_return_sequences=1, )[:,
                                               prompt.shape[-1]:]
                        model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
                        model_gen_str = model_gen_str.split('\n')[0].strip()
                        tmp_result_dict = dataset[j]
                        tmp_result_dict['model_response'] = model_gen_str
                        file.write(json.dumps(tmp_result_dict) + "\n")


if __name__ == "__main__":
    main()
