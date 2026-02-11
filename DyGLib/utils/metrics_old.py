import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score



def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor, mrr: bool = False):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    if not mrr:
        average_precision = average_precision_score(y_true=labels, y_score=predicts)
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        return {'average_precision': average_precision, 'roc_auc': roc_auc}
    else:
        # Calculate MRR
        # num_queries = len(labels)
        # ranks = []
        # for i in range(num_queries):
        #     # Sort indices by descending order of prediction scores
        #     sorted_indices = np.argsort(-predicts[i])
        #     # Find the rank of the first relevant item
        #     rank = np.where(sorted_indices == i)[0][0] + 1
        #     ranks.append(1 / rank)
        
        sorted_indices = np.argsort(-predicts)
        ranks = np.arange(len(predicts)) + 1
        relevant_ranks = ranks[labels[sorted_indices] == 1]
        mrr_value = np.mean(1 / relevant_ranks)
        metrics['mrr'] = mrr_value
        
        mrr_value = np.mean(ranks)
        return {"mrr": mrr_value}


# def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor, mrr: bool = False):
#     """
#     get metrics for the node classification task
#     :param predicts: Tensor, shape (num_samples, )
#     :param labels: Tensor, shape (num_samples, )
#     :return:
#         dictionary of metrics {'metric_name_1': metric_1, ...}
#     """
#     predicts = predicts.cpu().detach().numpy()
#     labels = labels.cpu().numpy()

#     if mrr:
#         # num_samples = len(labels)
#         # ranks = []
#         # for i in range(num_samples):
#         #     # Assuming binary classification, sort samples by prediction score
#         #     sorted_indices = np.argsort(-predicts)
#         #     # Find the rank of the positive sample
#         #     rank = np.where(sorted_indices == i)[0][0] + 1 if labels[i] == 1 else None
#         #     if rank is not None:
#         #         ranks.append(1 / rank)
        
#         # if ranks:
#         #     mrr_value = np.mean(ranks)
#         #     return {'mrr': mrr_value}
        
#         # Vectorized MRR calculation for binary classification
#         # Generate a mask for positive samples
#         positive_mask = labels == 1
#         # Get the indices that would sort the predicts array
#         sorted_indices = np.argsort(-predicts)
#         # Use advanced indexing to find the positions of positive samples in the sorted array
#         pos_sorted_indices = np.searchsorted(sorted_indices, np.where(positive_mask)[0])
#         # Compute ranks for positive samples
#         ranks = pos_sorted_indices + 1
#         # Compute reciprocals and take the mean to get MRR
#         mrr_value = np.mean(1 / ranks)
#         return {'mrr': mrr_value}
#     else:
#         roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
#         return {'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor, mrr: bool = False):
    """
    Get metrics for the node classification task.
    
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :param mrr: bool, whether to calculate Mean Reciprocal Rank (MRR)
    :return: dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    # Convert tensors to numpy arrays
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    if mrr:
        # Ensure there are positive samples
        if np.sum(labels) == 0:
            raise ValueError("No positive samples in labels for MRR calculation.")

        # Generate a mask for positive samples
        positive_mask = labels == 1
        # Get the indices of positive samples
        positive_indices = np.where(positive_mask)[0]
        # Get the sorted indices of the predictions
        sorted_indices = np.argsort(-predicts)
        # Initialize ranks array
        ranks = []

        for pos_index in positive_indices:
            # Find the rank of each positive sample
            rank = np.where(sorted_indices == pos_index)[0][0] + 1
            ranks.append(1 / rank)

        # Compute MRR
        mrr_value = np.mean(ranks)
        return {'mrr': mrr_value}

    else:
        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        return {'roc_auc': roc_auc}