#!/bin/bash
# Borrowed and modified from https://github.com/likenneth/honest_llama

python_interpreter="path to/envs/self/bin/python"

script_path="validate_2fold.py"

model_name="llama3.1_8B_instruct"

$python_interpreter $script_path --model_name "$model_name"
