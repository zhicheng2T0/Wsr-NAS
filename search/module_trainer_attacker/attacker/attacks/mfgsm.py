import torch
from torch import Tensor, nn


def mfgsm(
    model: nn.Module,
    criterion: nn.Module,
    image: Tensor,
    target: Tensor,
    data_mean: Tensor,
    data_std: Tensor,
    clip_eps: float,
    decay: float,
    n_repeats: int,
    model_type=True
):
    alpha = clip_eps / n_repeats
    image = image.detach().clone().to('cuda')
    target = target.detach().clone().to('cuda')
    momentum = torch.zeros_like(image).detach().to('cuda')
    adv_images = image.detach().clone()

    for i in range(n_repeats):
        adv_images.requires_grad = True
        if data_mean!=None:
            adv_images_norm = adv_images.sub(data_mean).div(data_std)
        else:
            adv_images_norm=adv_images
        if model_type:
            output = model(adv_images_norm)
        else:
            output,_ = model(adv_images_norm)
        loss = criterion(output, target)
        grad = torch.autograd.grad(loss, adv_images,
                                   retain_graph=False,
                                   create_graph=False)[0]

        # grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
        grad_norm = torch.norm(
            torch.flatten(grad, start_dim=1, end_dim=-1),
            p=1, dim=1
        )
        grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
        grad = grad + momentum * decay
        momentum = grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - image, min=-clip_eps, max=clip_eps)
        adv_images = torch.clamp(image + delta, min=0, max=1).detach()

    return adv_images
