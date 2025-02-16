import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import tokenized_dataset, get_llama_activations_bau, get_internlm_activations_bau, \
    get_mistral_activations_bau

base_path = os.path.dirname(__file__)

HF_NAMES = {
    'llama3.1_8B_instruct': 'path/to/original-hf-model/Meta-Llama-3.1-8B-Instruct',
    'llama3.1_70B_instruct': 'path/to/original-hf-model/Llama-3.1-8B-Instruct',
    'internlm2_5-20b-chat': 'path/to/original-hf-model/internlm2_5-20b-chat',
    'Mistral-Nemo-Instruct-2407': 'path/to/original-hf-model/Mistral-Nemo-Instruct-2407'
}

os.makedirs(os.path.join(base_path, 'features'), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='llama3.1_8B_instruct')
parser.add_argument('--dataset_name', type=str, default='situational_awareness')
parser.add_argument('--dataset_path', type=str, default=os.path.join(base_path, 'data'))
args = parser.parse_args()

model_name_or_path = HF_NAMES[args.model_name]

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                             device_map="auto", trust_remote_code=True)
device = "cuda"

dataset = load_dataset('json', data_files=os.path.join(args.dataset_path, args.dataset_name + '.json'))['train']
formatter = tokenized_dataset

print("Tokenizing prompts")
prompts, labels = formatter(dataset, tokenizer)

all_layer_wise_activations = []
all_head_wise_activations = []

print("Getting activations")
for prompt in tqdm(prompts):
    if 'llama' in args.model_name:
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
    elif 'internlm' in args.model_name:
        layer_wise_activations, head_wise_activations, _ = get_internlm_activations_bau(model, prompt, device)
    elif 'Mistral' in args.model_name:
        layer_wise_activations, head_wise_activations, _ = get_mistral_activations_bau(model, prompt, device)
    else:
        raise NotImplementedError
    all_layer_wise_activations.append(layer_wise_activations[:, -1, :].copy())
    all_head_wise_activations.append(head_wise_activations[:, -1, :].copy())

print("Saving labels")
np.save(os.path.join(base_path, 'features', f'{args.model_name}_{args.dataset_name}_labels.npy'), labels)

print("Saving layer wise activations")
np.save(os.path.join(base_path, 'features', f'{args.model_name}_{args.dataset_name}_layer_wise.npy'),
        all_layer_wise_activations)

print("Saving head wise activations")
np.save(os.path.join(base_path, 'features', f'{args.model_name}_{args.dataset_name}_head_wise.npy'),
        all_head_wise_activations)
