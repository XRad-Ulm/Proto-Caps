"""
Testing functions

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import os
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchmetrics import Dice
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from loss import xcaps_loss


def test(testmodel, data_loader, arguments):
    """
    Test model
    :param testmodel: model to be tested
    :param data_loader: data_loader used for testing
    :param arguments: parser arguments
    :return: [tested model, mal_accuracy, attribute_accuracies]
    """
    testmodel.eval()
    test_loss_total = 0
    num_attributes = next(iter(data_loader))[2].shape[1]
    correct_mal = 0
    correct_att = torch.zeros((num_attributes,))

    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, sampleID in data_loader:
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)
            pred_mal, pred_attr, x_recon, dists_to_protos = testmodel(x)
            pred_mal = torch.squeeze(pred_mal)

            loss, L_pred, L_recon, L_attr, L_cluster, L_sep = xcaps_loss(y_mal, pred_mal, y_mask, x_recon,
                                                                         arguments.lam_recon, y_attributes,
                                                                         pred_attr,
                                                                         1.0, epoch=0,
                                                                         dists_to_protos=dists_to_protos,
                                                                         sample_id=sampleID,
                                                                         idx_with_attri=sampleID,
                                                                         max_dist=testmodel.out_dim_caps,
                                                                         arguments=arguments)
            test_loss_total += loss.item() * x.size(0)

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

    test_loss_total /= len(data_loader.dataset)
    return test_loss_total, correct_mal / len(data_loader.dataset), \
        [correct_att[0] / len(data_loader.dataset),
         correct_att[1] / len(data_loader.dataset),
         correct_att[2] / len(data_loader.dataset),
         correct_att[3] / len(data_loader.dataset),
         correct_att[4] / len(data_loader.dataset),
         correct_att[5] / len(data_loader.dataset),
         correct_att[6] / len(data_loader.dataset),
         correct_att[7] / len(data_loader.dataset)]


def test_show(testmodel, data_loader, epoch,
              prototypefoldername):
    """
    Show results of inference
    :param testmodel: model to be tested
    :param test_loader: data loader used for testing
    :param epoch: number of epoch the model is being tested
    :param prototypefoldername: folder direction of prototypes
    """
    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, sampleID in data_loader:
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)
            pred_mal, pred_attr, _, dists_to_protos = testmodel(x)

            for sai in range(pred_mal.shape[0]):
                # select sample that is correctly/wrongly predicted, has a specific malignancy class, ...
                if abs(torch.argmax(pred_mal[sai]).item() - torch.argmax(y_mal[sai],
                                                                         dim=-1).item()) > 1 and torch.argmax(
                    y_mal[sai], dim=-1).item() < 2:
                    _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
                    min_proto_idx0 = [unravel_index(i, dists_to_protos[0][sai].shape) for i in min_proto_idx0]
                    _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
                    min_proto_idx1 = [unravel_index(i, dists_to_protos[1][sai].shape) for i in min_proto_idx1]
                    _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
                    min_proto_idx2 = [unravel_index(i, dists_to_protos[2][sai].shape) for i in min_proto_idx2]
                    _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
                    min_proto_idx3 = [unravel_index(i, dists_to_protos[3][sai].shape) for i in min_proto_idx3]
                    _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
                    min_proto_idx4 = [unravel_index(i, dists_to_protos[4][sai].shape) for i in min_proto_idx4]
                    _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
                    min_proto_idx5 = [unravel_index(i, dists_to_protos[5][sai].shape) for i in min_proto_idx5]
                    _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
                    min_proto_idx6 = [unravel_index(i, dists_to_protos[6][sai].shape) for i in min_proto_idx6]
                    _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
                    min_proto_idx7 = [unravel_index(i, dists_to_protos[7][sai].shape) for i in min_proto_idx7]
                    min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                             min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]

                    folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)
                    plt.figure(figsize=(15, 8))
                    plt.subplot(2, 5, 1)
                    plt.title("input")
                    plt.imshow(torch.squeeze(x[sai]).cpu(), cmap='gray')
                    plt.subplot(2, 5, 2)
                    plt.title("mask")
                    plt.imshow(torch.squeeze(y_mask[sai]).cpu(), cmap='gray')
                    counter = 3
                    plt.suptitle("mal score: " + str(torch.argmax(y_mal[sai], dim=-1).item() + 1) + ", pred: " + str(
                        torch.argmax(pred_mal[sai]).item() + 1))
                    cpsnames = ["sub", "is", "cal", "sph", "mar", "lob", "spic", "tex"]
                    for capsule_idx in range(len(min_proto_idx_allcaps)):
                        for image_name in os.listdir(folder_dir):
                            if image_name.startswith(
                                    "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                        min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                        min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                                proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                                protoimage = np.load(
                                    folder_dir + "/" + image_name)
                                plt.subplot(2, 5, counter)
                                counter += 1
                                plt.imshow(protoimage, cmap='gray')
                                plt.title(
                                    cpsnames[capsule_idx] + " (" + str(
                                        round(y_attributes[sai][capsule_idx].item(), 2)) + "," + str(
                                        round(pred_attr[sai, capsule_idx].item(), 2)) + "," + str(
                                        proto_attrsc_str[capsule_idx]) + ")")
                    plt.show()


def test_indepth_attripredCorr(testmodel, train_loader, test_loader, epoch,
                               prototypefoldername):
    """
    Test model regarding correlation of correctness of attribute and malignancy testing
    :param testmodel: model to be tested
    :param train_loader: data loader used for training
    :param test_loader: data loader used for testing
    :param epoch: number of epoch the model is being tested
    :param prototypefoldername: folder direction of prototypes
    """
    testmodel.eval()
    predp_attrip_correct_matrix = np.zeros(((8, 2, 2)))
    num_attributes = next(iter(test_loader))[2].shape[1]
    attricorr_train = torch.zeros((len(train_loader.dataset), num_attributes))
    attricorr_test = torch.zeros((len(test_loader.dataset), num_attributes))
    predcorr_train = torch.zeros((len(train_loader.dataset)))
    predcorr_test = torch.zeros((len(test_loader.dataset)))
    batch_num = -1
    batch_size = next(iter(test_loader))[0].shape[0]
    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, _ in test_loader:
            batch_num += 1
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)

            _, _, pred_mask, dists_to_protos = testmodel(x)

            _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
            min_proto_idx0 = [unravel_index(i, dists_to_protos[0][0].shape) for i in min_proto_idx0]
            _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
            min_proto_idx1 = [unravel_index(i, dists_to_protos[1][0].shape) for i in min_proto_idx1]
            _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
            min_proto_idx2 = [unravel_index(i, dists_to_protos[2][0].shape) for i in min_proto_idx2]
            _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
            min_proto_idx3 = [unravel_index(i, dists_to_protos[3][0].shape) for i in min_proto_idx3]
            _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
            min_proto_idx4 = [unravel_index(i, dists_to_protos[4][0].shape) for i in min_proto_idx4]
            _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
            min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
            _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
            min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
            _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
            min_proto_idx7 = [unravel_index(i, dists_to_protos[7][0].shape) for i in min_proto_idx7]
            min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                     min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]

            x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))

            attrLabelsSamples_protos = torch.zeros_like(y_attributes)
            folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)

            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):

                    x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]
                    for image_name in os.listdir(folder_dir):
                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_attrsc = torch.tensor(np.array(proto_attrsc_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_attrsc[capsule_idx]

            pred_p, _, _ = testmodel.forwardCapsule(x_ex=x_ex.to("cuda", dtype=torch.float))
            pred_p_correct_0 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_p1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_m1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) - 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_list = pred_p_correct_0 + pred_p_correct_p1 + pred_p_correct_m1
            predcorr_test[
            (batch_num * batch_size):(batch_num * batch_size + len(pred_p_correct_list))] = torch.from_numpy(
                pred_p_correct_list)

            y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
            y_attributes[:, 1] *= 3
            y_attributes[:, 2] *= 5
            y_attributes += 1
            attrLabelsSamples_protos[:, [0, 3, 4, 5, 6, 7]] *= 4
            attrLabelsSamples_protos[:, 1] *= 3
            attrLabelsSamples_protos[:, 2] *= 5
            attrLabelsSamples_protos += 1
            all_attri_correct = []
            for at in range(y_attributes.shape[1]):
                attri_p_correct_0 = np.rint(y_attributes[:, at].cpu().detach().numpy()) == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_p1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) + 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_m1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) - 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_list = attri_p_correct_0 + attri_p_correct_p1 + attri_p_correct_m1
                all_attri_correct.append(attri_p_correct_list)
                attricorr_test[(batch_num * batch_size):(batch_num * batch_size + len(attri_p_correct_list)),
                at] = torch.from_numpy(attri_p_correct_list)

            for sai in range(x.shape[0]):
                for attri in range(len(all_attri_correct)):
                    if (pred_p_correct_list[sai] == True) and (all_attri_correct[attri][sai] == True):
                        predp_attrip_correct_matrix[attri, 0, 0] += 1
                    if (pred_p_correct_list[sai] == False) and (all_attri_correct[attri][sai] == True):
                        predp_attrip_correct_matrix[attri, 0, 1] += 1
                    if (pred_p_correct_list[sai] == True) and (all_attri_correct[attri][sai] == False):
                        predp_attrip_correct_matrix[attri, 1, 0] += 1
                    if (pred_p_correct_list[sai] == False) and (all_attri_correct[attri][sai] == False):
                        predp_attrip_correct_matrix[attri, 1, 1] += 1
    cpsnames = ["sub", "is", "cal", "sph", "mar", "lob", "spic", "tex"]
    print("\t\t\t\t\t\t\t\tmal prediction")
    print("\t\t\t\t\t\t\t\tcorrect\tfalse")
    print("attri prediction\tcorrect")
    print("\t\t\t\t\tfalse")
    for attri in range(predp_attrip_correct_matrix.shape[0]):
        print(cpsnames[attri] + ":" + "\t\t\t\t\t\t\t" + str(predp_attrip_correct_matrix[attri][0, 0]) + "\t" + str(
            predp_attrip_correct_matrix[attri][0, 1]))
        print("\t\t\t\t\t\t\t\t" + str(predp_attrip_correct_matrix[attri][1, 0]) + "\t" + str(
            predp_attrip_correct_matrix[attri][1, 1]))
        print("FF/False attribute ratio: " + str(
            predp_attrip_correct_matrix[attri][1, 1] / np.sum(predp_attrip_correct_matrix[attri], axis=1)[1]))

    # train
    batch_num = -1
    batch_size = next(iter(train_loader))[0].shape[0]
    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, _ in train_loader:
            batch_num += 1
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)

            _, _, pred_mask, dists_to_protos = testmodel(x)

            _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
            min_proto_idx0 = [unravel_index(i, dists_to_protos[0][0].shape) for i in min_proto_idx0]
            _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
            min_proto_idx1 = [unravel_index(i, dists_to_protos[1][0].shape) for i in min_proto_idx1]
            _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
            min_proto_idx2 = [unravel_index(i, dists_to_protos[2][0].shape) for i in min_proto_idx2]
            _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
            min_proto_idx3 = [unravel_index(i, dists_to_protos[3][0].shape) for i in min_proto_idx3]
            _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
            min_proto_idx4 = [unravel_index(i, dists_to_protos[4][0].shape) for i in min_proto_idx4]
            _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
            min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
            _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
            min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
            _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
            min_proto_idx7 = [unravel_index(i, dists_to_protos[7][0].shape) for i in min_proto_idx7]
            min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                     min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]

            x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))

            attrLabelsSamples_protos = torch.zeros_like(y_attributes)
            folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)
            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):

                    x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]
                    for image_name in os.listdir(folder_dir):
                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_attrsc = torch.tensor(np.array(proto_attrsc_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_attrsc[capsule_idx]

            pred_p, _, _ = testmodel.forwardCapsule(x_ex=x_ex.to("cuda", dtype=torch.float))
            pred_p_correct_0 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_p1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_m1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) - 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_list = pred_p_correct_0 + pred_p_correct_p1 + pred_p_correct_m1
            predcorr_train[
            (batch_num * batch_size):(batch_num * batch_size + len(pred_p_correct_list))] = torch.from_numpy(
                pred_p_correct_list)

            y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
            y_attributes[:, 1] *= 3
            y_attributes[:, 2] *= 5
            y_attributes += 1
            attrLabelsSamples_protos[:, [0, 3, 4, 5, 6, 7]] *= 4
            attrLabelsSamples_protos[:, 1] *= 3
            attrLabelsSamples_protos[:, 2] *= 5
            attrLabelsSamples_protos += 1
            all_attri_correct = []
            for at in range(y_attributes.shape[1]):
                attri_p_correct_0 = np.rint(y_attributes[:, at].cpu().detach().numpy()) == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_p1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) + 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_m1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) - 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_list = attri_p_correct_0 + attri_p_correct_p1 + attri_p_correct_m1
                all_attri_correct.append(attri_p_correct_list)
                attricorr_train[(batch_num * batch_size):(batch_num * batch_size + len(attri_p_correct_list)),
                at] = torch.from_numpy(attri_p_correct_list)

    clflr = LogisticRegression(random_state=0).fit(attricorr_train, predcorr_train)
    print("LogReg train" + str(clflr.score(attricorr_train, predcorr_train)))
    print("LogReg test" + str(clflr.score(attricorr_test, predcorr_test)))
    clfrf = RandomForestClassifier(max_depth=2, random_state=0).fit(attricorr_train, predcorr_train)
    print("RandFor train" + str(clfrf.score(attricorr_train, predcorr_train)))
    print("RandFor test" + str(clfrf.score(attricorr_test, predcorr_test)))
    clfdt = DecisionTreeClassifier(random_state=0).fit(attricorr_train, predcorr_train)
    print("DecTree train" + str(clfdt.score(attricorr_train, predcorr_train)))
    print("DecTree test" + str(clfdt.score(attricorr_test, predcorr_test)))


def test_indepth(testmodel, data_loader, epoch,
                 prototypefoldername):
    """
    Test model and use prototype scores for accuracies
    :param testmodel: model to be tested
    :param data_loader: data_loader used for testing
    :param epoch: number of epoch the model is being tested
    :param prototypefoldername: folder direction of prototypes
    :return: [mal_accuracy, attribute_accuracies, dice score]
    """
    testmodel.eval()
    correct_mal = 0
    num_attributes = next(iter(data_loader))[2].shape[1]
    correct_attproto = torch.zeros((8,))
    dc_score = 0
    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, _ in data_loader:
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)

            _, _, pred_mask, dists_to_protos = testmodel(x)

            dice = Dice(average='micro').to("cuda")
            dc_score += (dice(pred_mask, y_mask.type(torch.int64)) * y_mask.shape[0])

            _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
            min_proto_idx0 = [unravel_index(i, dists_to_protos[0][0].shape) for i in min_proto_idx0]
            _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
            min_proto_idx1 = [unravel_index(i, dists_to_protos[1][0].shape) for i in min_proto_idx1]
            _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
            min_proto_idx2 = [unravel_index(i, dists_to_protos[2][0].shape) for i in min_proto_idx2]
            _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
            min_proto_idx3 = [unravel_index(i, dists_to_protos[3][0].shape) for i in min_proto_idx3]
            _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
            min_proto_idx4 = [unravel_index(i, dists_to_protos[4][0].shape) for i in min_proto_idx4]
            _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
            min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
            _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
            min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
            _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
            min_proto_idx7 = [unravel_index(i, dists_to_protos[7][0].shape) for i in min_proto_idx7]
            min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                     min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]

            x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))

            attrLabelsSamples_protos = torch.zeros_like(y_attributes)
            folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)
            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):

                    x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]
                    for image_name in os.listdir(folder_dir):
                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_attrsc = torch.tensor(np.array(proto_attrsc_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_attrsc[capsule_idx]

            pred_p, _, _ = testmodel.forwardCapsule(x_ex=x_ex.to("cuda", dtype=torch.float))

            mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                    np.argmax(pred_p.cpu().detach().numpy(), axis=1) + 1,
                                                    labels=[1, 2, 3, 4, 5])
            mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                     sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                     sum(np.diagonal(mal_confusion_matrix, offset=-1))
            correct_mal += mal_correct_within_one

            y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
            y_attributes[:, 1] *= 3
            y_attributes[:, 2] *= 5
            y_attributes += 1
            attrLabelsSamples_protos[:, [0, 3, 4, 5, 6, 7]] *= 4
            attrLabelsSamples_protos[:, 1] *= 3
            attrLabelsSamples_protos[:, 2] *= 5
            attrLabelsSamples_protos += 1
            for at in range(y_attributes.shape[1]):
                a_labels = [1, 2, 3, 4, 5]
                if num_attributes == 8:
                    if at == 1:
                        a_labels = [1, 2, 3, 4]
                    if at == 2:
                        a_labels = [1, 2, 3, 4, 5, 6]
                attr_confusion_matrix_proto = confusion_matrix(
                    np.rint(y_attributes[:, at].cpu().detach().numpy()),
                    np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()),
                    labels=a_labels)
                correct_attproto[at] += sum(np.diagonal(attr_confusion_matrix_proto, offset=0)) + \
                                        sum(np.diagonal(attr_confusion_matrix_proto, offset=1)) + \
                                        sum(np.diagonal(attr_confusion_matrix_proto, offset=-1))

    return correct_mal / len(data_loader.dataset), \
        [correct_attproto[0] / len(data_loader.dataset),
         correct_attproto[1] / len(data_loader.dataset),
         correct_attproto[2] / len(data_loader.dataset),
         correct_attproto[3] / len(data_loader.dataset),
         correct_attproto[4] / len(data_loader.dataset),
         correct_attproto[5] / len(data_loader.dataset),
         correct_attproto[6] / len(data_loader.dataset),
         correct_attproto[7] / len(data_loader.dataset)], \
           dc_score / len(data_loader.dataset)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append((index % dim).item())
        index = index // dim
    return tuple(reversed(out))
