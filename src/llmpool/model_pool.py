from typing import List
from llmpool.model import LLModel

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