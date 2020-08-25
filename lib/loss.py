import torch
from torch import nn
from lib.data_process.utils import show_volume


def _neg_loss(pred, gt, src):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.ge(0.5).float()
    neg_inds = gt.lt(0.5).float()
    neg_weights = torch.pow(gt, 2)
    print("neg_pred_id: {}, pos_pred_id:{}".format(torch.sum(pred.lt(0.5).float()), torch.sum(pred.ge(0.5).float())))
    # gt_cpu = neg_inds.detach().cpu().numpy()[0][0]

    pos_loss = -torch.log(pred + 0.0001) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred + 0.0001) * torch.pow(pred, 2) * neg_inds * neg_weights
    print("neg_id: {}, pos_id:{}".format(torch.sum(neg_inds), torch.sum(pos_inds)))

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    print("neg :{}, pos: {}".format(neg_loss, pos_loss))

    # if num_pos == 0:
    #     loss = loss - neg_loss
    # else:
    #     loss = loss - (pos_loss + neg_loss) / num_pos
    loss = pos_loss + neg_loss
    return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, src):
        return self.neg_loss(out, target, src)

class BceLoss(nn.Module):
    def __init__(self):
        super(BceLoss, self).__init__()
        self.neg_loss = nn.BCELoss()

    def forward(self, out, target):
        return self.neg_loss(out.view(-1), target.view(-1))

class ResidualLoss(nn.Module):
    def __init__(self):
        super(ResidualLoss, self).__init__()
        self.neg_loss = nn.L1Loss()

    def forward(self, out, target):
        return self.neg_loss(out, target)

