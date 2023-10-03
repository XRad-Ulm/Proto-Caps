"""
Push prototypes: update prototype vectors with samples of data_loader and save respective original image.

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import torch


def pushprotos(model_push, data_loader, idx_with_attri):
    model_push.eval()
    mindists_allcpsi = [torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1])) * torch.inf,
                        torch.ones((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1])) * torch.inf]
    if model_push.threeD:
        mindists_X0 = torch.zeros(
            (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X1 = torch.zeros(
            (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X2 = torch.zeros(
            (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X3 = torch.zeros(
            (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X4 = torch.zeros(
            (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X5 = torch.zeros(
            (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X6 = torch.zeros(
            (model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        mindists_X7 = torch.zeros(
            (model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
    else:
        mindists_X0 = torch.zeros(
            (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X1 = torch.zeros(
            (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X2 = torch.zeros(
            (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X3 = torch.zeros(
            (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X4 = torch.zeros(
            (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X5 = torch.zeros(
            (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X6 = torch.zeros(
            (model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X7 = torch.zeros(
            (model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
    mindists_X_allcpsi = [mindists_X0, mindists_X1, mindists_X2, mindists_X3, mindists_X4, mindists_X5, mindists_X6,
                          mindists_X7]
    mindists_sampledigis_allcps = [torch.zeros_like(model_push.protodigis0), torch.zeros_like(model_push.protodigis1),
                                   torch.zeros_like(model_push.protodigis2), torch.zeros_like(model_push.protodigis3),
                                   torch.zeros_like(model_push.protodigis4), torch.zeros_like(model_push.protodigis5),
                                   torch.zeros_like(model_push.protodigis6), torch.zeros_like(model_push.protodigis7)]
    mindists_alllabels = [torch.zeros((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 9)),
                                torch.zeros((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 9)),
                                torch.zeros((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 9)),
                                torch.zeros((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 9)),
                                torch.zeros((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 9)),
                                torch.zeros((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 9)),
                                torch.zeros((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], 9)),
                                torch.zeros((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], 9))]
    with torch.no_grad():
        for (x, y_mask, y_attributes, y_mal, sampleID) in data_loader:
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)

            pred_mal, pred_attr, x_recon, dists_to_protos = model_push(x)

            _, max_y_mal = torch.max(y_mal, dim=-1)
            for sai in range(x.shape[0]):
                if sampleID[sai] in idx_with_attri:
                    for capsule_idx in range(len(dists_to_protos)):
                        if torch.abs(y_attributes[sai, capsule_idx] - pred_attr[sai, capsule_idx]) < 0.25:
                            for protoidx in range(dists_to_protos[capsule_idx].shape[1]):
                                if capsule_idx in [0, 3, 4, 5, 6, 7]:
                                    if (((y_attributes[sai, capsule_idx] < 0.125) and protoidx == 0) or
                                            ((0.125 <= y_attributes[sai, capsule_idx] < 0.375) and protoidx == 1) or
                                            ((0.375 <= y_attributes[sai, capsule_idx] < 0.625) and protoidx == 2) or
                                            ((0.625 <= y_attributes[sai, capsule_idx] < 0.875) and protoidx == 3) or
                                            ((y_attributes[sai, capsule_idx] >= 0.875) and protoidx == 4)):
                                        for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                            if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                    mindists_allcpsi[capsule_idx][
                                                        protoidx, protoidx2]:
                                                mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                    dists_to_protos[capsule_idx][
                                                        sai, protoidx, protoidx2]
                                                mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                mindists_alllabels[capsule_idx][
                                                    protoidx, protoidx2] = torch.cat(
                                                    (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                    model_push.protodigis_list[capsule_idx][
                                                    protoidx, protoidx2, :].cpu()
                                elif capsule_idx == 1:
                                    if (((y_attributes[sai, capsule_idx] < 0.16) and protoidx == 0) or
                                            ((0.16 <= y_attributes[sai, capsule_idx] < 0.49) and protoidx == 1) or
                                            ((0.49 <= y_attributes[sai, capsule_idx] < 0.82) and protoidx == 2) or
                                            ((y_attributes[sai, capsule_idx] >= 0.82) and protoidx == 3)):
                                        for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                            if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                    mindists_allcpsi[capsule_idx][
                                                        protoidx, protoidx2]:
                                                mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                    dists_to_protos[capsule_idx][
                                                        sai, protoidx, protoidx2]
                                                mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                mindists_alllabels[capsule_idx][
                                                    protoidx, protoidx2] = torch.cat(
                                                    (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                    model_push.protodigis_list[capsule_idx][
                                                    protoidx, protoidx2, :].cpu()
                                elif capsule_idx == 2:
                                    if (((y_attributes[sai, capsule_idx] < 0.1) and protoidx == 0) or
                                            ((0.1 <= y_attributes[sai, capsule_idx] < 0.3) and protoidx == 1) or
                                            ((0.3 <= y_attributes[sai, capsule_idx] < 0.5) and protoidx == 2) or
                                            ((0.5 <= y_attributes[sai, capsule_idx] < 0.7) and protoidx == 3) or
                                            ((0.7 <= y_attributes[sai, capsule_idx] < 0.9) and protoidx == 4) or
                                            ((y_attributes[sai, capsule_idx] >= 0.9) and protoidx == 5)):
                                        for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                            if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                    mindists_allcpsi[capsule_idx][
                                                        protoidx, protoidx2]:
                                                mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                    dists_to_protos[capsule_idx][
                                                        sai, protoidx, protoidx2]
                                                mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                mindists_alllabels[capsule_idx][
                                                    protoidx, protoidx2] = torch.cat(
                                                    (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                    model_push.protodigis_list[capsule_idx][
                                                    protoidx, protoidx2, :].cpu()

    model_push.protodigis0.data.copy_(mindists_sampledigis_allcps[0].cuda())
    model_push.protodigis1.data.copy_(mindists_sampledigis_allcps[1].cuda())
    model_push.protodigis2.data.copy_(mindists_sampledigis_allcps[2].cuda())
    model_push.protodigis3.data.copy_(mindists_sampledigis_allcps[3].cuda())
    model_push.protodigis4.data.copy_(mindists_sampledigis_allcps[4].cuda())
    model_push.protodigis5.data.copy_(mindists_sampledigis_allcps[5].cuda())
    model_push.protodigis6.data.copy_(mindists_sampledigis_allcps[6].cuda())
    model_push.protodigis7.data.copy_(mindists_sampledigis_allcps[7].cuda())
    return model_push, mindists_X_allcpsi, mindists_alllabels
