import logging

import numpy as np
import torch

from torch.nn import functional as F

from module.estimator.utils import arch_matrix_to_graph


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, search_objective_weights,model, momentum, weight_decay,
                 arch_learning_rate, arch_weight_decay,
                 predictor, pred_learning_rate,
                 architecture_criterion=F.mse_loss,
                 predictor_criterion=F.mse_loss,
                 is_gae=False,
                 reconstruct_criterion=None,
                 preprocessor=None):
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay

        # models
        self.sow=search_objective_weights
        self.model = model
        self.predictor = predictor
        self.is_gae = is_gae
        self.preprocessor = preprocessor
        self.reconstruct_criterion = reconstruct_criterion
        if self.is_gae: assert self.reconstruct_criterion is not None

        # architecture optimization
        self.architecture_optimizer = torch.optim.Adam(
            self.model.arch_parameters(), lr=arch_learning_rate, betas=(0.5, 0.999)
        )
        self.architecture_criterion = architecture_criterion

        # predictor optimization
        self.predictor_optimizer = torch.optim.Adam(
            #self.predictor.predictor.parameters(), lr=pred_learning_rate, betas=(0.5, 0.999)
            self.predictor.parameters(), lr=pred_learning_rate, betas=(0.5, 0.999),
            weight_decay=arch_weight_decay
        )
        self.predictor_criterion = predictor_criterion

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model

    def predictor_step(self, x, y,y_ae_list, unsupervised=False):
        # clear prev gradient
        self.predictor_optimizer.zero_grad()
        if self.is_gae:
            # convert architecture parameters from matrix to graph
            adj_normal, opt_normal = arch_matrix_to_graph(x[0])
            adj_reduce, opt_reduce = arch_matrix_to_graph(x[1])
            # preprocess graphs
            if self.preprocessor is not None:
                processed_adj_normal, processed_opt_normal = self.preprocessor(adj=adj_normal, opt=opt_normal)
                processed_adj_reduce, processed_opt_reduce = self.preprocessor(adj=adj_reduce, opt=opt_reduce)
            else:
                processed_adj_normal, processed_opt_normal = adj_normal, opt_normal
                processed_adj_reduce, processed_opt_reduce = adj_reduce, opt_reduce
            # get output
            (opt_recon_normal, opt_recon_reduce), \
            (adj_recon_normal, adj_recon_reduce), \
            z, y_pred = self.predictor(
                opt=(processed_opt_normal, processed_opt_reduce),
                adj=(processed_adj_normal, processed_adj_reduce)
            )
            y_pred = y_pred.squeeze()
            # calculate loss
            loss = self.reconstruct_criterion(
                [opt_recon_normal, adj_recon_normal], [opt_normal, adj_normal]
            ) + self.reconstruct_criterion(
                [opt_recon_reduce, adj_recon_reduce], [opt_reduce, adj_reduce]
            )
            if not unsupervised:
                acc_mse = self.predictor_criterion(y_pred, y)
                loss *= 0.8
                loss += acc_mse
        else:
            if unsupervised: logging.warning('unsupervised is only available for auto-encoding')
            # get output
            #y_pred = self.predictor(x)
            out=self.predictor(x+torch.normal(0,0.01,x.shape).float().cuda())
            out=torch.squeeze(out,dim=0)
            #print(out.shape)
            # calculate loss
            #loss = self.predictor_criterion(y_pred, y)
            loss = self.predictor_criterion(out[:,0], y)
            for i in range(1,out.shape[1],1):
                loss=loss+self.predictor_criterion(out[:,i], y)
        # back-prop and optimization step
        loss.backward()
        self.predictor_optimizer.step()
        return out, loss

    def step(self):
        self.architecture_optimizer.zero_grad()
        loss = self._backward_step()
        loss.backward()
        self.architecture_optimizer.step()
        return loss

    def _backward_step(self):
        if self.is_gae:
            # convert architecture parameters from matrix to graph
            graphs = self.model.arch_weights(cat=False)
            adj_normal, opt_normal = arch_matrix_to_graph(graphs[0].unsqueeze(0))
            adj_reduce, opt_reduce = arch_matrix_to_graph(graphs[1].unsqueeze(0))
            # preprocess graphs
            if self.preprocessor is not None:
                processed_adj_normal, processed_opt_normal = self.preprocessor(adj=adj_normal, opt=opt_normal)
                processed_adj_reduce, processed_opt_reduce = self.preprocessor(adj=adj_reduce, opt=opt_reduce)
            else:
                processed_adj_normal, processed_opt_normal = adj_normal, opt_normal
                processed_adj_reduce, processed_opt_reduce = adj_reduce, opt_reduce
            # get output
            _, _, z, y_pred = self.predictor(
                opt=(processed_opt_normal, processed_opt_reduce),
                adj=(processed_adj_normal, processed_adj_reduce)
            )
            y_pred = y_pred.squeeze()
        else:
            #y_pred = self.predictor(self.model.arch_weights().unsqueeze(0))
            out=self.predictor(self.model.arch_weights().unsqueeze(0))
            out=torch.squeeze(out,dim=0)
        loss = self.sow[0]*self.architecture_criterion(out[:,0], torch.zeros_like(out[:,0]))
        for i in range(len(self.sow[1])):
            loss=loss+self.sow[1][0]*self.architecture_criterion(out[:,i+1], torch.zeros_like(out[:,i+1]))
        return loss

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta,  network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to('cuda')

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

