import os
import importlib.util

keyword_to_model_utils = {
    "claude": os.path.join("apis", "claude_utils.py"),
    "gpt": os.path.join("apis", "gpt_utils.py"),
    "internlm": os.path.join("apis", "internlm_utils.py"),
    "Llama": os.path.join("apis", "llama_utils.py"),
    "Mistral": os.path.join("apis", "mistral_utils.py"),
}

def select_model_utils_by_keyword(model_path):
    for keyword, utils_file in keyword_to_model_utils.items():
        if keyword in model_path:
            return utils_file
    raise ValueError(f"No matching model_utils found for model_path: {model_path}")

def dynamic_import(module_path):
    spec = importlib.util.spec_from_file_location("model_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_data_files(data_file, output_file):
    try:
        lines = open(data_file).readlines()

        old_lines = []
        if os.path.exists(output_file):
            old_lines = open(output_file, encoding='utf8').readlines()

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        f_out = open(output_file, 'w', encoding='utf8')
        for line in old_lines:
            f_out.write(line)
            f_out.flush()

        return lines, old_lines, f_out

    except Exception as e:
        raise RuntimeError(f"Error reading files: {e}")