from transformers import GenerationConfig

class LLModel:
    def __init__(self, name):
        self.name = name
        pass

    def stream_gen(self, prompt, gen_config: GenerationConfig, stopping_criteria=None):
        pass

    def batch_gen(self, prompts, gen_config: GenerationConfig, stopping_criteria=None):
        pass
