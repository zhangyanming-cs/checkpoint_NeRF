import torch

class NeRF(torch.nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        
    def forward(self, rays, render):
        """
        input rays.shape: [1920, 1080, 6]
        output rendered_image.shape: [1920, 1080, 4]
        """
        
        output = render(rays)

        return output