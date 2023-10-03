"""
Training functions

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from loss import xcaps_loss


def train_model(trainmodel, data_loader, arguments, epoch, optim, idx_with_attri):
    """
    Train model for one poch.
    :param trainmodel: model to be trained
    :param data_loader: data_loader used for training
    :param arguments: parser arguments
    :param epoch: current training epoch
    :param optim: training optimizer
    :param idx_with_attri: indices of samples used for attribute training
    :return: [trained model, mal_accuracy, attribute_accuracies]
    """
    trainmodel.train()
    torch.autograd.set_detect_anomaly(True)

    num_attributes = next(iter(data_loader))[2].shape[1]
    correct_mal = 0
    correct_att = torch.zeros((num_attributes,))

    for i, (x, y_mask, y_attributes, y_mal, sampleID) in enumerate(data_loader):
        x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
            y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)

        optim.zero_grad()
        pred_mal, pred_attr, x_recon, dists_to_protos = trainmodel(x)
        pred_mal = torch.squeeze(pred_mal)

        loss, L_pred, L_recon, L_attr, L_cluster, L_sep = xcaps_loss(y_mal, pred_mal, y_mask, x_recon,
                                                                     arguments.lam_recon,
                                                                     y_attributes,
                                                                     pred_attr,
                                                                     1.0, epoch=epoch,
                                                                     dists_to_protos=dists_to_protos,
                                                                     sample_id=sampleID,
                                                                     idx_with_attri=idx_with_attri,
                                                                     max_dist=trainmodel.out_dim_caps,
                                                                     arguments=arguments)

        loss.backward()
        optim.step()
        if len(pred_mal.shape) < 2:
            pred_mal = torch.unsqueeze(pred_mal, 0)
        mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                np.argmax(pred_mal.cpu().detach().numpy(), axis=1) + 1,
                                                labels=[1, 2, 3, 4, 5])
        mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                 sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                 sum(np.diagonal(mal_confusion_matrix, offset=-1))
        correct_mal += mal_correct_within_one

        y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
        y_attributes[:, 1] *= 3
        y_attributes[:, 2] *= 5
        y_attributes += 1
        pred_attr[:, [0, 3, 4, 5, 6, 7]] *= 4
        pred_attr[:, 1] *= 3
        pred_attr[:, 2] *= 5
        pred_attr += 1
        for at in range(y_attributes.shape[1]):
            a_labels = [1, 2, 3, 4, 5]
            if num_attributes == 8:
                if at == 1:
                    a_labels = [1, 2, 3, 4]
                if at == 2:
                    a_labels = [1, 2, 3, 4, 5, 6]
            attr_confusion_matrix = confusion_matrix(
                np.rint(y_attributes[:, at].cpu().detach().numpy()),
                np.rint(pred_attr[:, at].cpu().detach().numpy()),
                labels=a_labels)
            correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0)) + \
                               sum(np.diagonal(attr_confusion_matrix, offset=1)) + \
                               sum(np.diagonal(attr_confusion_matrix, offset=-1))

    return trainmodel, correct_mal / len(data_loader.dataset), \
        [correct_att[0] / len(data_loader.dataset),
         correct_att[1] / len(data_loader.dataset),
         correct_att[2] / len(data_loader.dataset),
         correct_att[3] / len(data_loader.dataset),
         correct_att[4] / len(data_loader.dataset),
         correct_att[5] / len(data_loader.dataset),
         correct_att[6] / len(data_loader.dataset),
         correct_att[7] / len(data_loader.dataset)]
