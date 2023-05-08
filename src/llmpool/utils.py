import yaml
import llmpool
import transformers

from llmpool import LLModelPool
from llmpool import LocalLLModel, LocalLoRALLModel

def load_yaml(filepath):
    yaml_dict = None

    with open(filepath, 'r') as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

    return yaml_dict

def instantiate_model(name, model_spec):
    model_type = model_spec['type']
    metadata = model_spec['metadata']
    gen_config = model_spec['generation_config']

    model_ckpt = {}
    load_config = {}

    model_type = eval(model_type)
    if isinstance(model_type, LocalLLModel) \
        or isinstance(model_type, LocalLoRALLModel):
        assert "model" in model_spec, "model ckpt config should be provided"
        model_ckpt = model_spec['model']

        if 'load' in model_spec:
            load_config = model_spec['load']

    if "model_cls" in load_config:
        load_config['model_cls'] = eval(load_config['model_cls'])

    if "tokenizer_cls" in load_config:
        load_config['tokenizer_cls'] = eval(load_config['tokenizer_cls'])

    model = model_type(
        name=name,
        **model_ckpt,
        **load_config
    )

    return model

def instantiate_models(filepath):
    model_pool = LLModelPool()

    model_specs = load_yaml(filepath)

    for name, model_spec in model_specs:
        model_pool.add_model(
            instantiate_model(name, model_spec)
        )

    return model_pool