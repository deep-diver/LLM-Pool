from transformers import GenerationConfig

class LLModelMetadata:
    def __init__(
        self, 
        thumb_path=None,
        thumb_xs_path=None,
    ):
        self.thumb_path = thumb_path
        self.thumb_xs_path = thumb_xs_path

class LLModel:
    def __init__(self, name, gen_config: GenerationConfig, metadata: LLModelMetadata=None):
        self.name = name
        self.metadata = metadata
        self.gen_config = gen_config
        pass

    def stream_gen(self, prompt, gen_config: GenerationConfig=None, stopping_criteria=None):
        pass

    def batch_gen(self, prompts, gen_config: GenerationConfig=None, stopping_criteria=None):
        pass