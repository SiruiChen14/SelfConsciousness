import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def initialize(model_name):
    # Initialize Model
    print("Initializing model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = model.eval()
    print("Finish initialization model.", flush=True)
    return tokenizer, model

def query(input_text, generator):
    tokenizer, model = generator
    response, history = model.chat(tokenizer, input_text, history=[])
    return response
