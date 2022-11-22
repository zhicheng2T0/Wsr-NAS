import torch
from torch import Tensor, nn


def fgsm_attack(
    model: nn.Module,
    criterion: nn.Module,
    image: Tensor,
    target: Tensor,
    data_mean: Tensor,
    data_std: Tensor,
    fgsm_step: float,
    model_type=False
):
    image = image.detach().clone().to('cuda')
    target = target.detach().clone().to('cuda')

    image.requires_grad = True
    if data_mean!=None:
        adv_images_norm = image.sub(data_mean).div(data_std)
    else:
        adv_images_norm = image
    if model_type == True:
        output = model(adv_images_norm)
    else:
        output,_ = model(adv_images_norm)
    loss = criterion(output, target)
    grad = torch.autograd.grad(loss, image,
                               retain_graph=False,
                               create_graph=False)[0]

    adv_images = image + fgsm_step * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images
