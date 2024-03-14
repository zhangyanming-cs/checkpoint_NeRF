from checkpoint_model import Checkpoint_for_NeRF
from torch_model import NeRF
from utils import INPUT_SHAPE, create_config, set_seed, loss_fn, init_render
import torch
import copy


if __name__ ==  "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()
    
    config = create_config()
    render = init_render(config, device)
    rays_1 = torch.randn(*INPUT_SHAPE, requires_grad=True, device=device)
    rays_2 = copy.deepcopy(rays_1).requires_grad_(True)
    rendered_image_checkpoint = Checkpoint_for_NeRF.apply(rays_1, render)
    rendered_image_torch = NeRF()(rays_2, render)
    assert torch.allclose(rendered_image_checkpoint, rendered_image_torch, atol=1e-4)
    print("The forward  pass is correct!")
    
    loss_1 = loss_fn(rendered_image_checkpoint, target=None)
    loss_2 = loss_fn(rendered_image_torch, target=None)
    loss_1.backward()
    loss_2.backward()
    assert torch.allclose(rays_1.grad, rays_2.grad, atol=1e-4)
    print("The backward pass is correct!")