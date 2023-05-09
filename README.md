# LLM-Pool

This simple project is to manage multiple LLM(Large Language Model)s in one place. Because there are too many fine-tuned LLMs, and it is hard to evaluate which one is bettern than others, it might be useful to test as many models as possible. Below is the two useful usecases that I had in mind when kicking off this project.

- compare generated text from different models side by side
- complete conversation in collaboration of different models

![](https://i.ibb.co/my2tf27/overview.png)

## Usecase

```python

from llmpool import LLModelPool
from llmpool import LocalLoRAModel
from llmpool import RemoteTxtGenIfLLModel

from transformers import AutoModelForCausalLM

model_pool = LLModelPool()
model_pool.add_models(
  # alpaca-lora 13b
  LocalLoRALLModel(
    "alpaca-lora-13b",
    "elinas/llama-13b-hf-transformers-4.29",
    "LLMs/Alpaca-LoRA-EvolInstruct-13B",
    model_cls=AutoModelForCausalLM
  ),
  
  RemoteTxtGenIfLLModel(
    "stable-vicuna-13b",
    "https://...:8080"
  ),
)

for model in model_pool:
  result = model.batch_gen(
    ["hello world"], 
    GenerationConfig(...)
  )
  print(result)
  
  _, stream_result = model.stream_gen(
    "hello world",
    GenerationConfig(...)
  )

  for ret in stream_results:
    if instanceof(model, LocalLoRALLModel) or \
      instanceof(model, LocalLLModel):
      print(ret, end='')
    else:
      print(ret.token.text, end='')

```

Alternatively, you can organize the model pool with yaml file

```python
from llmpool import instantiate_models

model_pool = instantiate_models('...yaml')
```

## Todo
- [ ] Add example notebooks
- [X] Yaml parser to add models to model pool
