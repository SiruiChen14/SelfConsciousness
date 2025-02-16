#!/bin/bash
# Borrowed and modified from https://github.com/likenneth/honest_llama

data_dir="../testing_set"

python_interpreter="path to/envs/self/bin/python"

script_path1="get_activation.py"
script_path2="validation/self_aware_heads.py"
script_path3="validation/self_aware_layers.py"
script_path4="validation/self_aware_for_each_head.py"

model_name="llama3.1_8B_instruct"

for json_file in "$data_dir"/*.json; do
    filename=$(basename "$json_file" .json)

    $python_interpreter $script_path1 --model_name "$model_name" --dataset_name "$filename"

    $python_interpreter $script_path2 --model_name "$model_name" --dataset_name "$filename"

    $python_interpreter $script_path3 --model_name "$model_name" --dataset_name "$filename"

done

$python_interpreter $script_path4 --model_name "$model_name"
