from text_generation import Client
from transformers import GenerationConfig

from llmpool.model import LLModel

class TxtGenIfLLModel(LLModel):
    def __init__(self, name, url, port, headers=None, cookies=None, timeout=10):
        super().__init__(name)

        if port is not None:
            port = f":{port}"

        self.client = Client(
            base_url=f"{url}{port}", headers=headers, cookies=cookies, timeout=timeout
        )

    def _stream_text(self, stream_results):
        for stream_result in stream_results:
            yield stream_result.token.text

    def stream_gen(self, prompt, gen_config: GenerationConfig, stopping_criteria=None):
        super().stream_gen(prompt, gen_config, stopping_criteria)

        stream = self.client.generate_stream(
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

        return None, self._stream_text(stream)

    def batch_gen(self, prompts, gen_config: GenerationConfig, stopping_criteria=None, best_of=None):
        super().batch_gen(prompts, gen_config, stopping_criteria)

        batch = self.client.generate(
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

        return batch.generated_text

    