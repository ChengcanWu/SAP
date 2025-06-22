import torch
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
    
def load_lora_model(model_path: str, v_register_layers: list):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    peft_model = get_peft_model(model, lora_config)

    for v in v_register_layers:
        layer = peft_model.base_model.model.model.layers[v]
        layer.mlp.down_proj.bias = torch.nn.Parameter(torch.zeros(layer.mlp.down_proj.out_features))

    lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n]
    bias_params = [p for n, p in peft_model.named_parameters() if 'mlp.down_proj.bias' in n]

    return peft_model, lora_params, bias_params

def load_optimizer(lora_params, bias_params, w_lr, v_lr):
    optimizer_lora = AdamW(lora_params, lr=w_lr)
    optimizer_bias = AdamW(bias_params, lr=v_lr)
    return optimizer_lora, optimizer_bias

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

    return tokenizer


