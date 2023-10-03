"""
Loading and preprocessing of LIDC-IDRI data.

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import pylidc as pl
import os.path
import h5py


def resize_data(imgs, masks, labels, out_dims):
    img_list = [];
    mask_list = [];
    label_list = []
    for i in trange(len(imgs)):
        for j in range(imgs[i].shape[2]):
            img = Image.fromarray(imgs[i][:, :, j])
            mask = Image.fromarray(masks[i][:, :, j])
            out_img = img.resize(out_dims)
            out_mask = mask.resize(out_dims)
            img_list.append(np.asarray(out_img, dtype=np.int16))
            mask_list.append(np.asarray(out_mask, dtype=np.uint8))
            label_list.append(labels[i])
    return np.asarray(img_list), np.asarray(mask_list), np.asarray(label_list)


def generateNoduleSplitFiles():
    """
    Generate data splits nodule and scan wise stratified regarding malignancy.
    """
    scan = pl.query(pl.Scan).all()
    nod_list = []
    nod_meanmal_list = []
    scan_i_list = []
    totalnod = 0
    for scani in range(len(scan)):
        nodules = scan[scani].cluster_annotations()
        print("This scan" + str(scani) + " has %d nodules." % len(nodules))
        for i, nod in enumerate(nodules):
            if len(nod) >= 3 and len(nod) <= 4:
                nod_mal = []
                for nod_anni in range(len(nod)):
                    nod_mal.append(nod[nod_anni].malignancy)
                if np.mean(nod_mal) != 3.0:
                    nod_list.append(totalnod)
                    totalnod += 1
                    nod_meanmal_list.append(np.mean(nod_mal))
                    scan_i_list.append(scani)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=12)
    foldsplit = skf.split(nod_list, np.rint(nod_meanmal_list), groups=scan_i_list)

    for i, (train_index, test_index) in enumerate(foldsplit):
        np.save('split' + str(i) + '_train', train_index)
        np.save('split' + str(i) + '_test', test_index)


def generateDatasetforSplit(splitnumber, threeD, resize_shape):
    """
    Generate and save data set for specified split. Preprocessing to image, mask and labels of nodule.
    :param splitnumber: number of split to be processed
    :param threeD: Bool, True: for cropping out volume, False: for cropping out center slice
    :param resize_shape: resize shape of slice/volume
    """
    split_train = np.load("split" + str(splitnumber) + "_train.npy")
    split_test = np.load("split" + str(splitnumber) + "_test.npy")
    scan = pl.query(pl.Scan).all()
    img_list_train, img_list_test = [], []
    mask_list_train, mask_list_test = [], []
    label_list_train, label_list_test = [], []
    totalnod = 0
    for scani in range(len(scan)):
        nodules = scan[scani].cluster_annotations()
        print("This scan " + str(scani) + " has %d nodules." % len(nodules))
        for i, nod in enumerate(nodules):
            if len(nod) >= 3 and len(nod) <= 4:
                nod_sub, nod_int, nod_calc, nod_spher, nod_mar, nod_lob, nod_spic, nod_tex = [], [], [], [], [], [], [], []
                nod_mal = []
                for nod_anni in range(len(nod)):
                    nod_sub.append(nod[nod_anni].subtlety)
                    nod_int.append(nod[nod_anni].internalStructure)
                    nod_calc.append(nod[nod_anni].calcification)
                    nod_spher.append(nod[nod_anni].sphericity)
                    nod_mar.append(nod[nod_anni].margin)
                    nod_lob.append(nod[nod_anni].lobulation)
                    nod_spic.append(nod[nod_anni].spiculation)
                    nod_tex.append(nod[nod_anni].texture)
                    nod_mal.append(nod[nod_anni].malignancy)
                if np.mean(nod_mal) != 3.0:
                    for nod_anni in range(len(nod)):
                        labels = [nod[nod_anni].subtlety, nod[nod_anni].internalStructure,
                                  nod[nod_anni].calcification,
                                  nod[nod_anni].sphericity, nod[nod_anni].margin, nod[nod_anni].lobulation,
                                  nod[nod_anni].spiculation, nod[nod_anni].texture, nod[nod_anni].malignancy,
                                  np.mean(nod_sub), np.std(nod_sub), np.mean(nod_int), np.std(nod_int),
                                  np.mean(nod_calc), np.std(nod_calc),
                                  np.mean(nod_spher), np.std(nod_spher), np.mean(nod_mar), np.std(nod_mar),
                                  np.mean(nod_lob), np.std(nod_lob),
                                  np.mean(nod_spic), np.std(nod_spic), np.mean(nod_tex), np.std(nod_tex),
                                  np.mean(nod_mal), np.std(nod_mal)]

                        mask = nod[nod_anni].boolean_mask()
                        bbox = nod[nod_anni].bbox()
                        vol = nod[nod_anni].scan.to_volume()
                        centroid = nod[nod_anni].centroid.astype(int)
                        entire_mask = np.zeros_like(vol)
                        entire_mask[bbox] = mask
                        if not threeD:
                            for slice_i in range(mask.shape[-1]):
                                img_PIL = Image.fromarray(vol[bbox][:, :, slice_i])
                                mask_PIL = Image.fromarray(mask[:, :, slice_i])
                                out_img = img_PIL.resize(resize_shape)
                                out_mask = mask_PIL.resize(resize_shape)
                                img_nparray = np.asarray(out_img, dtype=np.int16)
                                mask_nparray = np.asarray(out_mask, dtype=np.uint8)
                                if totalnod in split_train:
                                    img_list_train.append(img_nparray)
                                    mask_list_train.append(mask_nparray)
                                    label_list_train.append(labels)
                                elif totalnod in split_test:
                                    img_list_test.append(img_nparray)
                                    mask_list_test.append(mask_nparray)
                                    label_list_test.append(labels)
                        else:
                            cropout_cube_size = 48
                            cropout_cube_size_half = cropout_cube_size / 2
                            cropout_border = np.array(
                                [[0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]]])
                            for d in range(3):
                                if int(centroid[d] - cropout_cube_size_half) < 0 or int(
                                        centroid[d] + cropout_cube_size_half) > vol.shape[d]:
                                    if int(centroid[d] - cropout_cube_size_half) < 0:
                                        cropout_border[d, 1] = cropout_cube_size
                                    else:
                                        cropout_border[d, 0] = vol.shape[d] - cropout_cube_size
                                else:
                                    cropout_border[d, 0] = int(centroid[d] - cropout_cube_size_half)
                                    cropout_border[d, 1] = int(centroid[d] + cropout_cube_size_half)
                            new_img = vol[cropout_border[0, 0]:cropout_border[0, 1],
                                      cropout_border[1, 0]:cropout_border[1, 1],
                                      cropout_border[2, 0]:cropout_border[2, 1]]
                            new_mask = entire_mask[cropout_border[0, 0]:cropout_border[0, 1],
                                       cropout_border[1, 0]:cropout_border[1, 1],
                                       cropout_border[2, 0]:cropout_border[2, 1]]
                            if totalnod in split_train:
                                img_list_train.append(new_img)
                                mask_list_train.append(new_mask)
                                label_list_train.append(labels)
                            elif totalnod in split_test:
                                img_list_test.append(new_img)
                                mask_list_test.append(new_mask)
                                label_list_test.append(labels)

                            if (totalnod > 0) and (totalnod % 20 == 0):
                                if len(img_list_train) > 0:
                                    img_train, mask_train, label_train = np.asarray(img_list_train), np.asarray(
                                        mask_list_train), np.asarray(
                                        label_list_train)
                                    if not os.path.isfile("3d_train_split" + str(splitnumber) + ".h5"):
                                        with h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'w') as h5f:
                                            h5f.create_dataset("img", shape=(
                                                0, img_train.shape[1], img_train.shape[2], img_train.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, img_train.shape[1], img_train.shape[2],
                                                                         img_train.shape[3]))
                                            h5f.create_dataset("mask", shape=(
                                                0, mask_train.shape[1], mask_train.shape[2], mask_train.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, mask_train.shape[1], mask_train.shape[2],
                                                                         mask_train.shape[3]))
                                            h5f.create_dataset("label", shape=(
                                                0, label_train.shape[1]), chunks=True,
                                                               maxshape=(None, label_train.shape[1]))
                                    with h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'a') as h5f:
                                        h5f["img"].resize((h5f["img"].shape[0] + img_train.shape[0]), axis=0)
                                        h5f["img"][-img_train.shape[0]:] = img_train
                                        h5f["mask"].resize((h5f["mask"].shape[0] + mask_train.shape[0]), axis=0)
                                        h5f["mask"][-mask_train.shape[0]:] = mask_train
                                        h5f["label"].resize((h5f["label"].shape[0] + label_train.shape[0]), axis=0)
                                        h5f["label"][-label_train.shape[0]:] = label_train
                                if len(img_list_test) > 0:
                                    img_test, mask_test, label_test = np.asarray(img_list_test), np.asarray(
                                        mask_list_test), np.asarray(
                                        label_list_test)
                                    if not os.path.isfile("3d_test_split" + str(splitnumber) + ".h5"):
                                        with h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'w') as h5f:
                                            h5f.create_dataset("img", shape=(
                                                0, img_test.shape[1], img_test.shape[2], img_test.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, img_test.shape[1], img_test.shape[2],
                                                                         img_test.shape[3]))
                                            h5f.create_dataset("mask", shape=(
                                                0, mask_test.shape[1], mask_test.shape[2], mask_test.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, mask_test.shape[1], mask_test.shape[2],
                                                                         mask_test.shape[3]))
                                            h5f.create_dataset("label", shape=(
                                                0, label_test.shape[1]), chunks=True,
                                                               maxshape=(None, label_test.shape[1]))
                                    with h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'a') as h5f:
                                        h5f["img"].resize((h5f["img"].shape[0] + img_test.shape[0]), axis=0)
                                        h5f["img"][-img_test.shape[0]:] = img_test
                                        h5f["mask"].resize((h5f["mask"].shape[0] + mask_test.shape[0]), axis=0)
                                        h5f["mask"][-mask_test.shape[0]:] = mask_test
                                        h5f["label"].resize((h5f["label"].shape[0] + label_test.shape[0]), axis=0)
                                        h5f["label"][-label_test.shape[0]:] = label_test

                                img_list_train, img_list_test = [], []
                                mask_list_train, mask_list_test = [], []
                                label_list_train, label_list_test = [], []
                    totalnod += 1

    img_train, mask_train, label_train = np.asarray(img_list_train), np.asarray(mask_list_train), np.asarray(
        label_list_train)
    img_test, mask_test, label_test = np.asarray(img_list_test), np.asarray(mask_list_test), np.asarray(
        label_list_test)
    if not threeD:
        np.save("img_train_split" + str(splitnumber), img_train)
        np.save("mask_train_split" + str(splitnumber), mask_train)
        np.save("label_train_split" + str(splitnumber), label_train)
        np.save("img_test_split" + str(splitnumber), img_test)
        np.save("mask_test_split" + str(splitnumber), mask_test)
        np.save("label_test_split" + str(splitnumber), label_test)
    else:
        if len(img_list_train) > 0:
            img_train, mask_train, label_train = np.asarray(img_list_train), np.asarray(
                mask_list_train), np.asarray(
                label_list_train)
            with h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'a') as h5f:
                h5f["img"].resize((h5f["img"].shape[0] + img_train.shape[0]), axis=0)
                h5f["img"][-img_train.shape[0]:] = img_train
                h5f["mask"].resize((h5f["mask"].shape[0] + mask_train.shape[0]), axis=0)
                h5f["mask"][-mask_train.shape[0]:] = mask_train
                h5f["label"].resize((h5f["label"].shape[0] + label_train.shape[0]), axis=0)
                h5f["label"][-label_train.shape[0]:] = label_train
        if len(img_list_test) > 0:
            img_test, mask_test, label_test = np.asarray(img_list_test), np.asarray(
                mask_list_test), np.asarray(
                label_list_test)
            with h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'a') as h5f:
                h5f["img"].resize((h5f["img"].shape[0] + img_test.shape[0]), axis=0)
                h5f["img"][-img_test.shape[0]:] = img_test
                h5f["mask"].resize((h5f["mask"].shape[0] + mask_test.shape[0]), axis=0)
                h5f["mask"][-mask_test.shape[0]:] = mask_test
                h5f["label"].resize((h5f["label"].shape[0] + label_test.shape[0]), axis=0)
                h5f["label"][-label_test.shape[0]:] = label_test


def load_lidc(batch_size,
              splitnumber,
              threeD,
              resize_shape):
    """
    Generates DataLoader.
    :param batch_size: batch size of returning DataLoader
    :param splitnumber: number of split to be processed
    :param threeD: Bool, True: for cropping out volume, False: for cropping out center slice
    :param resize_shape: resize shape of slice/volume
    :return: Dataloaders of train, validation and test data
    """
    path = "split" + str(splitnumber) + "_train.npy"
    if not os.path.isfile(path):
        print("Split dataset nodule wise.")
        generateNoduleSplitFiles()
    if not threeD:
        path = "img_train_split" + str(splitnumber) + ".npy"
        if not os.path.isfile(path):
            print("Create dataset for split " + str(splitnumber) + ".")
            generateDatasetforSplit(splitnumber, threeD, resize_shape)

        img_train = np.load("img_train_split" + str(splitnumber) + ".npy")
        mask_train = np.load("mask_train_split" + str(splitnumber) + ".npy")
        img_test = np.load("img_test_split" + str(splitnumber) + ".npy")
        mask_test = np.load("mask_test_split" + str(splitnumber) + ".npy")
        label_train = np.load("label_train_split" + str(splitnumber) + ".npy")
        label_test = np.load("label_test_split" + str(splitnumber) + ".npy")
    else:
        path = "3d_train_split" + str(splitnumber) + ".h5"
        if not os.path.isfile(path):
            print("Create 3d dataset for split " + str(splitnumber) + ".")
            generateDatasetforSplit(splitnumber, threeD, resize_shape)
        h5f = h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'r')
        img_train = np.array(h5f['img'])
        mask_train = np.array(h5f['mask'])
        label_train = np.array(h5f['label'])
        h5f = h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'r')
        img_test = np.array(h5f['img'])
        mask_test = np.array(h5f['mask'])
        label_test = np.array(h5f['label'])

        print(img_train.shape)
        print(mask_train.shape)
        print(label_train.shape)
        print(img_test.shape)
        print(mask_test.shape)
        print(label_test.shape)

    split_train, split_val = train_test_split(torch.arange(img_train.shape[0]), test_size=0.1,
                                              stratify=label_train[:, 25])
    train_imgs = img_train[split_train]
    train_masks = mask_train[split_train]
    train_labels = label_train[split_train]
    val_imgs = img_train[split_val]
    val_masks = mask_train[split_val]
    val_labels = label_train[split_val]
    test_imgs = img_test
    test_masks = mask_test
    test_labels = label_test

    fin_train_dataset = []
    counter = 0
    for i in range(train_imgs.shape[0]):
        attr_label = np.asarray([(train_labels[i, 9] - 1) / 4.0,
                                 (train_labels[i, 11] - 1) / 3.0,
                                 (train_labels[i, 13] - 1) / 5.0,
                                 (train_labels[i, 15] - 1) / 4.0,
                                 (train_labels[i, 17] - 1) / 4.0,
                                 (train_labels[i, 19] - 1) / 4.0,
                                 (train_labels[i, 21] - 1) / 4.0,
                                 (train_labels[i, 23] - 1) / 4.0])
        mal_label = np.asarray(
            distribution_label(x=[1., 2., 3., 4., 5.], mu=train_labels[i, 25], sig=train_labels[i, 26]))

        fin_train_dataset.append(
            [np.expand_dims(train_imgs[i], axis=0), np.expand_dims(train_masks[i], axis=0), attr_label, mal_label,
             counter])
        counter += 1
    train_loader = torch.utils.data.DataLoader(fin_train_dataset, batch_size=batch_size, shuffle=True)

    fin_val_dataset = []
    counter = 0
    for i in range(val_imgs.shape[0]):
        attr_label = np.asarray([(val_labels[i, 9] - 1) / 4.0,
                                 (val_labels[i, 11] - 1) / 3.0,
                                 (val_labels[i, 13] - 1) / 5.0,
                                 (val_labels[i, 15] - 1) / 4.0,
                                 (val_labels[i, 17] - 1) / 4.0,
                                 (val_labels[i, 19] - 1) / 4.0,
                                 (val_labels[i, 21] - 1) / 4.0,
                                 (val_labels[i, 23] - 1) / 4.0])
        mal_label = np.asarray(distribution_label(x=[1., 2., 3., 4., 5.], mu=val_labels[i, 25], sig=val_labels[i, 26]))
        fin_val_dataset.append(
            [np.expand_dims(val_imgs[i], axis=0), np.expand_dims(val_masks[i], axis=0), attr_label, mal_label, counter])
        counter += 1
    val_loader = torch.utils.data.DataLoader(fin_val_dataset, batch_size=batch_size)

    fin_test_dataset = []
    counter = 0
    for i in range(test_imgs.shape[0]):
        attr_label = np.asarray([(test_labels[i, 9] - 1) / 4.0,
                                 (test_labels[i, 11] - 1) / 3.0,
                                 (test_labels[i, 13] - 1) / 5.0,
                                 (test_labels[i, 15] - 1) / 4.0,
                                 (test_labels[i, 17] - 1) / 4.0,
                                 (test_labels[i, 19] - 1) / 4.0,
                                 (test_labels[i, 21] - 1) / 4.0,
                                 (test_labels[i, 23] - 1) / 4.0])
        mal_label = np.asarray(
            distribution_label(x=[1., 2., 3., 4., 5.], mu=test_labels[i, 25], sig=test_labels[i, 26]))
        fin_test_dataset.append(
            [np.expand_dims(test_imgs[i], axis=0), np.expand_dims(test_masks[i], axis=0), attr_label, mal_label,
             counter])
        counter += 1
    test_loader = torch.utils.data.DataLoader(fin_test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def distribution_label(x, mu, sig):
    """
    Generates distribution.
    :param x: sample points
    :param mu: mean
    :param sig: standard deviation
    :return: list of values representing distribution
    """
    if sig < .05:
        sig = .05
    d0 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[0] - mu) / (2 * sig))
    d1 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[1] - mu) / (2 * sig))
    d2 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[2] - mu) / (2 * sig))
    d3 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[3] - mu) / (2 * sig))
    d4 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[4] - mu) / (2 * sig))

    return [d0, d1, d2, d3, d4]
