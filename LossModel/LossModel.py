import torch
import torch.nn.functional as F
from Config.TrainConfig import TrainConfig
from typing import Dict
from math import exp, log


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(neg_loss.shape, pos_loss.shape)
    # print("neg_loss,pos_loss", neg_loss.mean(), pos_loss.mean())
    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    # seq_len = y_pred.shape[2]
    # t = y_pred[0, 0, :, :]
    # res = 0.0
    # for i in range(seq_len):
    #     for j in range(seq_len):
    #         if t[i, j] > -1000:
    #             res += exp(float(t[i, j].cpu().data))
    #             # print("{}".format(float(t[i, j].cpu().data)), end=" ")
    #     # print("\n")
    # print("结束嘞", res)
    # print("结束嘞", log(res))
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_pred, y_true)
    return loss


class LossModel(torch.nn.Module):
    def __init__(self, conf: TrainConfig):
        super().__init__()
        self.conf = conf

    def forward(self, ipt: Dict, model_output: Dict):
        """ """
        logits = model_output["logits"]
        label = ipt["label"].to(logits.device)
        return loss_fun(label, logits)
