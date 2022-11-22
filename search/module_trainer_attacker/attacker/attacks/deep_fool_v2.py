import torch as torch
from torch.autograd.gradcheck import zero_gradients


def deep_fool_attack_v2(model, image, data_mean, data_std, num_classes=10, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    w = torch.zeros_like(image)
    r_tot = torch.zeros_like(image)
    image = image.sub(data_mean).div(data_std).detach().clone()

    with torch.no_grad():
        prediction_nat = model(image)
    ranking = prediction_nat.argsort(dim=-1, descending=True)
    label = ranking[:, 0]

    pert_image = image.detach().clone()
    pert_image.requires_grad = True

    # print(prediction_adv.size())
    # print(ranking.size())
    # print(ranking)
    # exit()
    # fs_list = [prediction_adv[0, ranking[k]] for k in range(num_classes)]

    for step in range(max_iter):

        prediction_adv = model(pert_image)
        k_i = torch.argmax(prediction_adv, dim=-1)
        if (k_i != label).all():
            break

        pert = torch.tensor(float('inf'))
        grad_orig = torch.autograd.grad(prediction_adv[:, ranking[:, 0]], pert_image,
                                        retain_graph=True,
                                        create_graph=False)[0]

        for k in range(1, num_classes):
            zero_gradients(pert_image)
            cur_grad = torch.autograd.grad(prediction_adv[:, ranking[:, k]], pert_image,
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

        pert_image = image + (1 + overshoot) * r_tot
        pert_image = pert_image.detach().clone()
        pert_image.requires_grad = True

    # r_tot = (1 + overshoot) * r_tot
    # return r_tot, loop_i, label, k_i, pert_image

    pert_image = pert_image * data_std + data_mean
    pert_image = torch.clamp(pert_image, min=0, max=255).detach()
    return pert_image
