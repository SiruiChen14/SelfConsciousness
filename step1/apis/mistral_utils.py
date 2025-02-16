import torch
import transformers

def initialize(model_name):
  # Initialize Model
  print("Initializing model...", flush=True)
  pipeline = transformers.pipeline(
      "text-generation",
      model=model_name,
      model_kwargs={"torch_dtype": torch.bfloat16},
      device_map="auto",
  )
  print("Finish initialization model.", flush=True)
  return pipeline

def internlm_api(input_text, generator):
    messages = [{"role": "user", "content": input_text}]
    outputs = generator(
        messages,
        max_new_tokens=256,
        )
    response = outputs[0]["generated_text"][-1]['content']
    return response
