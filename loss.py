"""
Loss function combining attribute, target and prototype learning.

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import torch
from torch import nn


def xcaps_loss(y, y_pred, x, x_recon, lam_recon, attr_gt, attr_pred, lam_attr, epoch,
               dists_to_protos, sample_id, idx_with_attri, max_dist, arguments):
    L_recon = nn.MSELoss()(x_recon, x)

    L_pred = nn.KLDivLoss(reduction="batchmean")(y_pred, y)

    batchidx_with_attri = []
    for i in range(len(sample_id)):
        if sample_id[i] in idx_with_attri:
            batchidx_with_attri.append(i)
    L_attr = 0.0
    for i in range(attr_gt.shape[-1]):
        L_attr += nn.MSELoss()(attr_pred[batchidx_with_attri, i], attr_gt[batchidx_with_attri, i])

    L_sep = 0.0
    L_cluster_allmean = 0.0
    L_cluster_allcpsi = 0.0

    if epoch < arguments.warmup:
        total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr
    else:
        if len(batchidx_with_attri) > 0:
            for capsule_idx in range(len(dists_to_protos)):
                if capsule_idx in [0, 3, 4, 5, 6, 7]:
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.125).nonzero()
                    idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.125) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.375)).nonzero()
                    idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.375) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.625)).nonzero()
                    idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.625) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.875)).nonzero()
                    idx4 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.875).nonzero()
                    idxs = [idx0, idx1, idx2, idx3, idx4]
                    L_cluster = 0

                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.min(torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]),dim=-1)[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist, selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],indices=[idx0,idx1,idx2,idx3,idx4])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 4)

                elif capsule_idx == 1:
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.16).nonzero()
                    idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.16) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.49)).nonzero()
                    idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.49) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.82)).nonzero()
                    idx3 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.82).nonzero()
                    idxs = [idx0, idx1, idx2, idx3]
                    L_cluster = 0

                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.min(torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]),dim=-1)[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist,
                                          selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                          indices=[idx0, idx1, idx2, idx3])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 3)

                elif capsule_idx == 2:
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.1).nonzero()
                    idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.1) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.3)).nonzero()
                    idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.3) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.5)).nonzero()
                    idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.5) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.7)).nonzero()
                    idx4 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.7) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.9)).nonzero()
                    idx5 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.9).nonzero()
                    idxs = [idx0, idx1, idx2, idx3, idx4, idx5]
                    L_cluster = 0

                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.min(torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]),dim=-1)[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist,
                                          selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                          indices=[idx0, idx1, idx2, idx3, idx4, idx5])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 5)

        L_sep /= len(dists_to_protos)
        L_cluster_allmean = L_cluster_allcpsi / len(dists_to_protos)
        total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + (1 / 8) * (
                L_cluster_allmean + 0.1 * L_sep)
    return total_loss, L_pred, L_recon, L_attr, L_cluster_allmean, L_sep

def sep_loss(max_dist, selected_dists, indices):
        num_classes = len(indices)
        ranges = [list(range(1,num_classes))]
        for i in range(num_classes-2):
            ranges.append(list(range(0,num_classes-(num_classes-1-i)))+list(range(i+2,num_classes)))
        ranges.append(list(range(0,num_classes-1)))
        loss_temp = []
        for i in range(num_classes):
            loss_temp.append(torch.min(torch.maximum(torch.zeros_like(
                max_dist - selected_dists[indices[i],ranges[i]]),
                (max_dist - selected_dists[indices[i],ranges[i]])),
                dim=-1)[0])
        return torch.sum(torch.cat(loss_temp, dim=0))
