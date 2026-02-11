import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import sys


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    mask_pos = labels == 1
    mask_neg = labels == 0

    pos_pred = torch.tensor(predicts[mask_pos])
    neg_pred = torch.tensor(predicts[mask_neg])
    # print("Positive predictions: ",pos_pred)
    # print(len(pos_pred))


    ranks = torch.sum(pos_pred.unsqueeze(1) <= neg_pred.unsqueeze(0), dim=1) + 1
    # print(ranks)
    # print(len(ranks))

    # Calculate Mean Reciprocal Rank (MRR)
    reciprocal_ranks = 1.0 / ranks
    mrr = reciprocal_ranks.mean()

    # sys.exit()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc,'mrr':mrr}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
