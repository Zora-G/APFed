import random

import numpy as np
import torch


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def model_to_vector(model):
    values = []
    for value in model.state_dict().values():
        if torch.is_floating_point(value):
            values.append(value.detach().cpu().reshape(-1).float())
    if len(values) == 0:
        return torch.tensor([])
    return torch.cat(values)

def vector_to_model(vector, model):
    """Copy a flattened floating-point vector into a model state dict."""
    state = model.state_dict()
    offset = 0
    for name, value in state.items():
        if not torch.is_floating_point(value):
            continue
        count = value.numel()
        part = vector[offset:offset + count].view_as(value).to(value.device, value.dtype)
        state[name] = part
        offset += count
    if offset != int(vector.numel()):
        raise ValueError('vector length does not match model parameters')
    model.load_state_dict(state)
    return model

def get_model_update(local_model, global_model):
    """Return the gradient direction g_i = theta_global-theta_local."""
    return model_to_vector(global_model) - model_to_vector(local_model)
