import torch
from torch import nn
import torch.nn.functional as F

class Handle_Croess_Entropy_Loss(nn.Module):
    def __init__(self, ignore_index=-1, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        if ignore_index != -1:
            self.weight[ignore_index] = 0.0

    def forward2(self, output, target):
        # (nb_sample, nb_class), (nb_sample)
        _softmax = F.softmax(output, dim=1)
        neg_log_softmax = -torch.log(_softmax)

        nb_sample = 0
        loss = []

        for y_hat, y in zip(neg_log_softmax, target):
            if y == self.ignore_index:
                continue
            nb_sample += 1
            y_hat_c = self.weight[y]*y_hat[y] if self.weight is not None else y_hat[y]
            loss.append(y_hat_c)
        # print(loss)
        loss = sum(loss) / nb_sample

        return loss

    def forward(self, output, target):
        # (nb_sample, nb_class), (nb_sample)
        # 这里对input所有元素求exp
        exp = torch.exp(output)
        # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
        tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
        # 在exp第一维求和，这是softmax的分母
        tmp2 = exp.sum(1)
        # softmax公式：ei / sum(ej)
        _softmax = tmp1 / tmp2
        # cross-entropy公式： -yi * log(pi)
        # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
        # 公式中的pi就是softmax的结果
        neg_log_softmax = -torch.log(_softmax) # (nb_sample)
        if self.weight is not None:
            weight = self.weight.gather(0, target).squeeze()
            neg_log_softmax = weight * neg_log_softmax

        nb_sample = (target != self.ignore_index).long().sum()
        loss = neg_log_softmax.sum() / nb_sample
        
        return loss
