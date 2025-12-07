import torch
import numpy as np
from torchmetrics import MeanAbsoluteError
from sklearn.metrics import average_precision_score

def get_metric_fn(cfg):
    if cfg.Train.metric == "accuracy":
        return batch_accuracy
    if cfg.Train.metric == "average-precision":
        return eval_ap
    if cfg.Train.metric == "mae":
        return MeanAbsoluteError().to(cfg.Device)
    
def eval_ap(y_pred,y_true):
    """ Code taken form LRGB repo. 
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    ap_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            ap_list.append(ap)
    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')
    return  sum(ap_list) / len(ap_list)

def batch_accuracy(X:torch.Tensor,y:torch.Tensor) -> float:
    return torch.sum(torch.argmax(X, dim=-1) == y)/len(y)