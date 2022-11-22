import torch
from torch.nn import DataParallel


class StandardConstraint(object):

    def __init__(self, model, criterion, optimizer, grad_clip=5., backbone=None):
        super(StandardConstraint, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.backbone = backbone

        self._zero_tenor_cuda = torch.tensor(0.).to('cuda')

    def _apply_constraint(self, x, target):
        # model make prediction
        if isinstance(self.model, DataParallel):
            self.model.module.use_weak()
        else:
            self.model.use_weak()
        pred_m = self.model(x)
        loss_m = self.criterion(pred_m, target)

        # backbone make prediction
        with torch.no_grad():
            if self.backbone is not None:
                pred_b = self.backbone(x)
            else:
                if isinstance(self.model, DataParallel):
                    self.model.module.use_base()
                else:
                    self.model.use_base()
                pred_b = self.model(x)
            loss_b = self.criterion(pred_b, target)

        loss_const = loss_m - loss_b

        if loss_const > 0.:
            loss_const = torch.max(loss_const, self._zero_tenor_cuda) ** 2
            loss_const.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # update model
            self.optimizer.step()

        return loss_const

    def __call__(self, x, target):
        return self._apply_constraint(x, target)
