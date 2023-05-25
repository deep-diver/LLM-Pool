import yaml
from typing import List

from transformers import GenerationConfig

import llmpool
from llmpool.model import LLModel
from llmpool.local_model import LocalLLModel

class LLModelIter:
    def __init__(self, model_pool):
        self._models = model_pool.models
        self._pool_size = len(self._models)
        self._current_index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._current_index < self._pool_size:
            model = self._models[self._current_index]
            self._current_index += 1
            return model
        
        raise StopIteration
    
class LLModelPool:
    def __init__(self):
        self.models = {}

    def add_model(self, model: LLModel):
        self.models[model.name] = model

    def add_models(self, models: List[LLModel]):
        for model in models:
            self.models[model.name] = model

    def get_model(self, name) -> LLModel:
        return self.models[name]

    def __iter__(self):
        return LLModelIter(self)
    
    @classmethod
    def get_default_gen_config(cls):
        return {
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 128,
            "penalty_p": 0.5
        }

    @classmethod
    def instantiate_model(cls, name, model_spec):
        model_type = model_spec['type']
        load_config = model_spec['load']
        metadata = model_spec['metadata']

        model_ckpt = {}
        if 'generation_config' in model_spec:
            gen_config = cls.get_default_gen_config()
        else:
            gen_config = model_spec['generation_config']        

        model_type = eval(model_type)
        if isinstance(model_type, LocalLLModel):
            assert "model" in model_spec, "model ckpt config should be provided"
            model_ckpt = model_spec['model']

        if "model_cls" in load_config:
            load_config['model_cls'] = eval(load_config['model_cls'])

        if "tokenizer_cls" in load_config:
            load_config['tokenizer_cls'] = eval(load_config['tokenizer_cls'])

        model = model_type(
            name=name,
            gen_config=GenerationConfig(
                **gen_config
            ),
            **model_ckpt,
            **load_config
        )

        return model

    @classmethod
    def from_yaml(cls, filepath):
        model_pool = cls()

        model_specs = cls.load_yaml(filepath)

        for name, model_spec in model_specs:
            model_pool.add_model(
                cls.instantiate_model(name, model_spec)
            )

        return model_pool
    
    @classmethod
    def load_yaml(cls, filepath):
        yaml_dict = None

        with open(filepath, 'r') as file:
            yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

        return yaml_dict
