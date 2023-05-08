# LLM-Pool

![](https://i.ibb.co/my2tf27/overview.png)

## Usecase

```python

from llmpool import LLModelPool
from llmpool import LocalLoRAModel
from llmpool import RemoteTxtGenIfLLModel

from transformers import AutoModelForCausalLM

model_pool = LLModelPool()
model_pool.add_model(
  # alpaca-lora 13b
  LocalLoRALLModel(
    "alpaca-lora-13b",
    "elinas/llama-13b-hf-transformers-4.29",
    "LLMs/Alpaca-LoRA-EvolInstruct-13B",
    model_cls=AutoModelForCausalLM
  ),
  
  RemoteTxtGenIfLLModel(
    "stable-vicuna-13b",
    "API_URL"
  )
)

for model in model_pool:
  batch_result = model.batch_gen(
    ["hello world"], 
    GenerationConfig(...)
  )
  
  _, stream_result = model.stream_gen(
    "hello world",
    GenerationConfig(...)
  )
  for text in stream_result:
    print(text, end='')

```
