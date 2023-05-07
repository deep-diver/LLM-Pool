from llmpool.model import LLModel

class LLModelPool:
    def __init__(self):
        self.models = {}

    def add_model(self, model: LLModel):
        self.models[model.name] = model

    def get_model(self, name) -> LLModel:
        return self.models[name]
