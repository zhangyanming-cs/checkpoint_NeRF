import torch
from utils import MAX_BATCHSIZE

class Checkpoint_for_NeRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rays, render):
        """
        input rays.shape: [1920, 1080, 6]
        output rendered_image.shape: [1920, 1080, 4]
        """
        SHAPE = rays.shape
        max_threshold_batchsize = MAX_BATCHSIZE
        rays_num = rays.shape[0] * rays.shape[1]
        
        rays = rays.reshape(-1, rays.shape[-1])

        ctx.input_tensors = rays
        ctx.rays_num = rays_num
        ctx.max_threshold_batchsize = max_threshold_batchsize
        ctx.render = render

        output_images = []
        
        for i in range(0, rays_num, max_threshold_batchsize):
            batch = rays[i:i+max_threshold_batchsize]
            
            with torch.no_grad():
                output = render(batch)
            output_images.append(output)

        output_images = torch.cat(output_images, dim=0).reshape(*SHAPE[0:2], -1).requires_grad_(True)

        return output_images

    @staticmethod
    def backward(ctx, output_grad):
        SHAPE = output_grad.shape
        rays = ctx.input_tensors.detach().requires_grad_(True)
        output_grad = output_grad.reshape(-1, output_grad.shape[-1])

        input_grades = []
        for i in range(0, ctx.rays_num, ctx.max_threshold_batchsize):
            with torch.enable_grad():
                batch = rays[i:i+ctx.max_threshold_batchsize]
                grad_batch = output_grad[i:i+ctx.max_threshold_batchsize]
                output = ctx.render(batch)
                
            input_grade = torch.autograd.grad(output, batch, grad_outputs=grad_batch)
            input_grades.append(input_grade[0])
        input_grades = torch.cat(input_grades, dim=0)
        return input_grades.reshape(*SHAPE[0:2], -1), None