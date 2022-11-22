import torch as torch
from torch.autograd.gradcheck import zero_gradients


def _deep_fool_attack(model, image, data_mean, data_std, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    pert = torch.zeros_like(image)
    w = torch.zeros_like(image)
    r_tot = torch.zeros_like(image)
    image = image.sub(data_mean).div(data_std).detach().clone()
    clip_eps = 0.031

    with torch.no_grad():
        prediction_nat = model(image[None]).flatten()
    ranking = prediction_nat.argsort(descending=True)
    label = ranking[0]

    pert_image = image.detach().clone()
    pert_image.requires_grad = True

    loop_i = 0

    prediction_adv = model(pert_image[None])
    # print(prediction_adv.size())
    # print(ranking.size())
    # print(ranking)
    # exit()
    # fs_list = [prediction_adv[0, ranking[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = torch.tensor(float('inf'))
        grad_orig = torch.autograd.grad(prediction_adv[0, ranking[0]], pert_image,
                                        retain_graph=True,
                                        create_graph=False)[0]

        for k in range(1, num_classes):
            zero_gradients(pert_image)
            cur_grad = torch.autograd.grad(prediction_adv[0, ranking[k]], pert_image,
                                           retain_graph=True,
                                           create_graph=False)[0]

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (prediction_adv[0, ranking[k]] - prediction_adv[0, ranking[0]]).detach().clone()

            pert_k = torch.abs(f_k) / torch.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / torch.norm(w)
        r_tot = r_tot + r_i
        pert = (1 + overshoot) * r_tot

        pert_image = image + pert
        pert_image = pert_image.detach().clone()
        pert_image.requires_grad = True

        prediction_adv = model(pert_image[None])
        k_i = torch.argmax(prediction_adv.flatten())

        loop_i += 1

    # print('r_tot, %.4f, %.4f, %.4f' % (r_tot.min(), r_tot.mean(), r_tot.max()))

    pert = torch.clamp(pert, min=-clip_eps, max=clip_eps)
    pert_image = image + pert
    pert_image = pert_image * data_std + data_mean
    # print('pert_image, %.4f, %.4f, %.4f' % (pert_image.min(), pert_image.mean(), pert_image.max()))

    pert_image = torch.clamp(pert_image, min=0., max=1.).detach()
    return pert_image


def deep_fool_attack(model, image, data_mean, data_std, num_classes=10, overshoot=0.02, max_iter=50):
    pert_image = []
    for img in image:
        pert_img = _deep_fool_attack(model=model, image=img, data_mean=data_mean, data_std=data_std,
                                     num_classes=num_classes, overshoot=overshoot, max_iter=max_iter)
        pert_image.append(pert_img)
    pert_image = torch.stack(pert_image)
    return pert_image
