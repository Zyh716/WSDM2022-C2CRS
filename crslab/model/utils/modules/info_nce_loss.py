# @Time    :   2022/1/1
# @Author  :   Yuanhang Zhou
# @email   :   sdzyh002@gmail.com

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def masked_info_nce_loss(features, sample_ids, co_occur_mask, device, temperature):
    """ from https://github.com/sthalles/SimCLR.git
    features: (nb_sample, dim)
    sample_ids: (nb_sample) =  [0, 0, 1, 
                                0, 0, 1, 2]
    co_occur_mask: (nb_sample) =  [0, 0, 0, 
                                   1, 1, 1, 1]
    """
    # sample id of features = [0, 1, .., bs-1, 0, 1, .., bs-1]

    labels = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)).float() # (nb_sample, nb_sample)
    co_occur_mask = (co_occur_mask.unsqueeze(0) == co_occur_mask.unsqueeze(1)).float() # (nb_sample, nb_sample)

    features = F.normalize(features, dim=1) # (nb_sample, dim)

    similarity_matrix = torch.matmul(features, features.T) # (nb_sample, nb_sample)
    assert similarity_matrix.shape == (features.shape[0], features.shape[0])
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device) # (nb_sample, nb_sample), eye
    labels = labels[~mask].view(labels.shape[0], -1) # (nb_sample, nb_sample-1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (nb_sample, nb_sample-1)
    co_occur_mask = co_occur_mask[~mask].view(co_occur_mask.shape[0], -1) # (nb_sample, nb_sample-1)
    assert similarity_matrix.shape == labels.shape
    assert co_occur_mask.shape == labels.shape

    labels, similarity_matrix, co_occur_mask = split_multi_positives(labels, similarity_matrix, co_occur_mask, device)  # (nb_sample++, nb_sample-1)

    # select and combine multiple positives
    # print('similarity_matrix.shape = ', similarity_matrix.shape)
    # print('labels.shape = ', labels.shape)
    # print('similarity_matrix[labels.bool()].shape = ', similarity_matrix[labels.bool()].shape)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # (nb_sample++, 1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # (nb_sample++, nb_sample-2)

    pos_co_occur_mask = co_occur_mask[labels.bool()].view(labels.shape[0], -1) # (nb_sample++, 1)
    neg_co_occur_mask = co_occur_mask[~labels.bool()].view(labels.shape[0], -1) # (nb_sample++, nb_sample-2)

    logits = torch.cat([positives, negatives], dim=1) # (nb_sample++, nb_sample-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device) # (nb_sample++)
    logits_mask = torch.cat([pos_co_occur_mask, neg_co_occur_mask], dim=1) # (nb_sample++, nb_sample-1)

    logits = logits * logits_mask / temperature
    return logits, labels


def split_multi_positives(labels, similarity_matrix, co_occur_mask, device):
    '''labels = [
                 [0, 0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                 ...
                ]
        (nb_sample, nb_sample-1)

        what we want is:
        labels = [
                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                 ...
                ]
        (nb_sample++, nb_sample-1)

        same for similarity_matrix and co_occur_mask
    '''
    dim1, dim2 = labels.shape

    new_labels = []
    new_sim_matrix = []
    new_co_occur_mask = []

    for label_line, sim_line, co_occur_mask_line in zip(labels, similarity_matrix, co_occur_mask):
        if int(torch.sum(label_line)) > 1:
            for idx in np.nonzero(label_line):
                idx = int(idx)

                new_label_line = torch.zeros((1, dim2), dtype=torch.long).to(device)
                new_label_line[0][idx] = 1
                new_labels.append(new_label_line)

                new_sim_matrix.append(sim_line.unsqueeze(0))
                new_co_occur_mask.append(co_occur_mask_line.unsqueeze(0))
        else:
            new_labels.append(label_line.unsqueeze(0))
            new_sim_matrix.append(sim_line.unsqueeze(0))
            new_co_occur_mask.append(co_occur_mask_line.unsqueeze(0))
    new_labels = torch.cat(new_labels, dim=0)
    new_sim_matrix = torch.cat(new_sim_matrix, dim=0)
    new_co_occur_mask = torch.cat(new_co_occur_mask, dim=0)

    return new_labels, new_sim_matrix, new_co_occur_mask


def info_nce_loss(features, bs, n_views, device, temperature):
    """ from https://github.com/sthalles/SimCLR.git
        features = (n_views*bs, dim)
        n_views: rgcn, transformer's view on user_rep
    """
    labels = torch.cat([torch.arange(bs) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    assert similarity_matrix.shape == (n_views * bs, n_views * bs)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels
