import torch
import argparse

INPUT_SHAPE = [1920, 1080, 6]
OUTPUT_SHAPE = [1920, 1080, 4]
MAX_BATCHSIZE = 200000


def init_render(config, device = "cuda"):
    def render(rays):
        output = rays * 2 + 5
        return output
    
    render_module = torch.nn.Linear(INPUT_SHAPE[-1], 4 if config.uselinear else 6, device=device)
    torch.nn.init.kaiming_uniform_(render_module.weight, a=1)
    render_module.weight.requires_grad_(True)
    render_module.bias.requires_grad_(True)
    
    if config.uselinear:
        ret = render_module
    elif config.usefunc:
        ret = render
    else:
        raise NotImplementedError("Please specify your own render function")
    
    return ret



def set_seed(seed = 42):
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def loss_fn(input, target = None):
    if target is None:
        target = torch.zeros_like(input, device=input.device)
    return torch.sum((input - target))

def create_config():
    parser = argparse.ArgumentParser(description='CheckPoint NeRF')
    parser.add_argument('--uselinear', type=int, default=1)
    parser.add_argument('--usefunc', type=int, default=0)
    args = parser.parse_args()
    return args