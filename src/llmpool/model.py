from transformers import GenerationConfig

class LLModel:
    def __init__(self, name):
        self.name = name
        pass

    def stream_gen(self, gen_config: GenerationConfig):
        pass

    def batch_gen(self, gen_config: GenerationConfig):
        pass
