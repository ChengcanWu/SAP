import torch

def set_optimizer(optimizer, requires_grad):
    for param in optimizer.param_groups[0]['params']:
        param.requires_grad = requires_grad

def zero_params(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.data.zero_()

def get_lora_grads(model):
    lora_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            lora_grads[name] = param.grad
    return lora_grads

def get_lora_params(model, norm_only = False):
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name.split('.') or 'lora_B' in name.split('.'):
            lora_params[name] = param.clone()
    if norm_only:
        return flatten_parameters(lora_params).norm(p=2, dim=-1, keepdim=True)
    else:
        return lora_params

def safe_normalize(vector, eps=1e-8):
    norm = vector.norm(p=2, dim=-1, keepdim=True)
    return vector / (norm + eps)

def flatten_parameters(params_dict, return_shape = False):
    shapes = {name: param.shape for name, param in params_dict.items()}
    flat_vector = torch.cat([param.flatten() for param in params_dict.values()])
    if return_shape:
        return flat_vector, shapes
    else:
        return flat_vector

def unflatten_parameters(flat_vector, shapes):
    params_dict = {}
    start = 0
    for name, shape in shapes.items():
        end = start + torch.prod(torch.tensor(shape)).item()
        params_dict[name] = flat_vector[start:end].reshape(shape)
        start = end
    return params_dict

def normalize_dict(paras_dict):
    paras_vector, shapes = flatten_parameters(paras_dict, return_shape=True)
    return unflatten_parameters(safe_normalize(paras_vector), shapes)

def merge_lora_parameters(
    model: torch.nn.Module,
    updated_lora_params: dict[str, torch.Tensor], 
    update_rate: float):
    for name, param in model.named_parameters():
        if name in updated_lora_params:
            param.data.copy_(
                param.data + update_rate * updated_lora_params[name]
            )

