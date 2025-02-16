from openai import OpenAI
import time


def initialize(model_name):
  return model_name

def query(input_text, generator):
  client = OpenAI(
    api_key= "your own key",
  )
  retries=3
  retry_cnt = 0
  backoff_time = 30
  while retry_cnt < retries:
    try:
      response = client.chat.completions.create(
          model=generator, 
          messages=[
              {"role": "user", "content": input_text}
          ]
      )
      return response.choices[0].message.content
    except Exception as e:
      print(e)
      time.sleep(backoff_time)
      backoff_time *= 1.5
      retry_cnt += 1
  return None


