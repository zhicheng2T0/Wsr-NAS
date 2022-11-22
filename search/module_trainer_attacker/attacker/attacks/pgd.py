import torch
from torch import Tensor, nn


def pgd_attack(
    model: nn.Module,
    criterion: nn.Module,
    image: Tensor,
    target: Tensor,
    data_mean: Tensor,
    data_std: Tensor,
    clip_eps: float,
    fgsm_step: float,
    n_repeats: int,
    pert_at=None,
    random_init=False,
    case=False
):
    all_pert = {}
    if pert_at is None:
        pert_at = []

    # init pert
    if random_init:
        pert = (torch.rand_like(image) * 2 - 1) * clip_eps  # get uniform random noise
    else:
        pert = torch.zeros_like(image).requires_grad_(False)  # zero init

    for step in range(n_repeats):
        # apply pert to image
        image_pert_with_grad = (image + pert).clamp(0., 1.).detach().requires_grad_(True)
        image_pert = image_pert_with_grad.sub(data_mean).div(data_std)  # normalization
        # cal grad
        if case==False:
            output,_ = model(image_pert)  # forward prop
        else:
            output = model(image_pert)  # forward prop
        loss = criterion(output, target)  # loss cal
        grad = torch.autograd.grad(loss, image_pert_with_grad,
                                   retain_graph=False, create_graph=False)[0]
        # update pert
        pert += fgsm_step * torch.sign(grad.detach())  # take a FGSM step
        pert.clamp_(-clip_eps, clip_eps)  # clip
        # save pert
        if step+1 in pert_at:
            all_pert[step+1] = pert.detach().clone()

    if n_repeats not in all_pert:
        all_pert[n_repeats] = pert.detach().clone()

    return all_pert
