import argparse
import json
from utils import dynamic_import, read_data_files, select_model_utils_by_keyword

parser = argparse.ArgumentParser()
parser.add_argument('--active_model', default=None, type=str, required=True)
parser.add_argument('--data_file', default=None, type=str, help="A file that contains instructions (one instruction per line)")
parser.add_argument('--output_file', default="./output.jsonl", type=str, help="Output file.")
args = parser.parse_args()

if __name__ == "__main__":
    filename, model_name, outname = args.data_file, args.active_model, args.output_file

    lines, old_lines, f_out = read_data_files(filename, outname)
    old_lines_len = len(old_lines)

    model_utils_path = select_model_utils_by_keyword(args.active_model)

    model_utils = dynamic_import(model_utils_path)
    initialize = model_utils.initialize
    query = model_utils.query

    generator = initialize(model_name)

    data_list = []
    for index in range(old_lines_len, len(lines)):
        line = lines[index]
        item = json.loads(line)
        data_list.append(item)

    instruction_begin = "Answer the following question with only the most correct option and no extra content.\n"
    instruction_end = "\nAnswer: "

    for index, data in enumerate(data_list):
        input_text = instruction_begin + data["question"] + instruction_end

        response = query(input_text, generator)

        print(f"======={index}=======", flush=True)
        print(f"query: {input_text}\n", flush=True)
        print(f"response: {response}\n", flush=True)

        data["model_response"] = response
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        f_out.flush()

    f_out.close()