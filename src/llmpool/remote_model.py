from text_generation import Client
from transformers import GenerationConfig

from llmpool.model import LLModel

class TextGenInferenceLLModel(LLModel):
    def __init__(self, name, url, headers=None, cookies=None, timeout=10):
        super().__init__(name)

        self.client = Client(
            base_url=url, headers=headers, cookies=cookies, timeout=timeout
        )

    def stream_gen(self, prompt, gen_config: GenerationConfig, stopping_criteria=None):
        super().stream_gen(prompt, gen_config, stopping_criteria)

        stream = client.generate_stream(
            prompt,
            do_sample=gen_config.do_sample,
            max_new_tokens=gen_config.max_new_tokens,
            repetition_penalty=gen_config.repetition_penalty,
            return_full_text=False,
            seed=None,
            stop_sequences=None,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            truncate=None,
            typical_p=gen_config.typical_p,
            watermark=False,
        )

        return None, stream

    def batch_gen(self, prompts, gen_config: GenerationConfig, stopping_criteria=None, best_of=None):
        super().batch_gen(prompts, gen_config, stopping_criteria)

        batch = client.generate(
            prompts,
            do_sample=gen_config.do_sample,
            max_new_tokens=gen_config.max_new_tokens,
            best_of=best_of,
            repetition_penalty=gen_config.repetition_penalty,
            return_full_text=False,
            seed=None,
            stop_sequences=None,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            truncate=None,
            typical_p=gen_config.typical_p,
            watermark=False,
        )

        return batch

    