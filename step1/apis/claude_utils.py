import anthropic
import time


def initialize(model_name):
  return model_name

def query(input_text, generator):
  client = anthropic.Anthropic(
  api_key="your own key",
)
  retries=3
  retry_cnt = 0
  backoff_time = 30
  while retry_cnt < retries:
    try:
      response = client.messages.create(
        model=generator,
        max_tokens=128,
        messages=[
            {"role": "user", "content": input_text}
            ]
        )
      return response.content[0].text
    except Exception as e:
      print(e)
      time.sleep(backoff_time)
      backoff_time *= 1.5
      retry_cnt += 1
  return None
