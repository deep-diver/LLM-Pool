from threading import Thread

from peft import PeftModel
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

from llmpool.model import LLModel

class LocalLLModel(LLModel):
    def stream_gen(self, prompts, gen_config: GenerationConfig, stopping_criteria=None):
        supert().stream_gen(prompts, gen_config, stopping_criteria)

        model_inputs = self._build_model_inputs(prompt)
        streamer = self._build_streamer()
        gen_kwargs = self._build_gen_kwargs(
            model_inputs, gen_config, streamer, stopping_criteria
        )

        t = Thread(target=self.model.generate, kwargs=gen_kwargs)
        return thread, streamer

    def _build_gen_kwargs(self, model_inputs, gen_config, streamer, stopping_criteria):
        gen_kwargs = dict(
            model_inputs,
            streamer=streamer,
            stopping_criteria=stopping_criteria
        )
        gen_kwargs.update(gen_config.__dict__.copy())
        return gen_kwargs 

    def _build_model_inputs(self, prompt, return_token_type_ids):
        model_inputs = self.tokenizer(
            [prompt], 
            return_tensors="pt",
            return_token_type_ids=False
        ).to(self.device)
        return model_inputs

    def _build_streamer(self, timeout=20., skip_prompt=True, skip_special_tokens=True):
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=timeout, 
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens
        )
        return streamer

    def batch_gen(self, prompts, gen_config: GenerationConfig, stopping_criteria=None):
        super().batch_gen(prompts, gen_config, stopping_criteria)

        if len(prompts) == 1:
            encoding = tokenizer(prompts, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self.device)
            generated_id = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
            )
            decoded = tokenizer.batch_decode(
                generated_id, skip_prompt=True, skip_special_tokens=True
            )
            return decoded
        else:
            encodings = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            generated_ids = model.generate(
                **encodings,
                generation_config=generation_config,
            )

            decoded = tokenizer.batch_decode(
                generated_ids, skip_prompt=True, skip_special_tokens=True
            )
            del encodings, generated_ids
            torch.cuda.empty_cache()
            return decoded              

class LocalCausalLLModel(LocalLLModel):
    def __init__(
        self, name, base, device='cuda', 
        load_in_8bit=True, apply_bettertransformer=False
    ):
        super().__init__(name)

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base,
            load_in_8bit=load_in_8bit,
            device_map="auto",
        )
        
        if apply_bettertransformer:
            self.model = BetterTransformer.transform(self.model)

        if not load_in_8bit:
            self.model.half()

class LocalLoRACausalLLModel(LocalCausalLLModel):
    def __init__(
        self, name, base, ckpt, device='cuda', 
        load_in_8bit=True, apply_bettertransformer=False
    ):
        super().__init__(
            name, base, device, load_in_8bit, 
            apply_bettertransformer=apply_bettertransformer)

        self.model = PeftModel.from_pretrained(
            self.model, ckpt, 
            device_map={'': 0}
        )
