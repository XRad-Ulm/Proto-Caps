"""
Pytorch implementation of Proto-Caps in paper
    Interpretable Medical Image Classification using Prototype Learning and Privileged Information .

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import sys
import torch
import datetime
import numpy as np
from models import ProtoCapsNet
from data_loader import load_lidc
from push import pushprotos
from train import train_model
from test import test, test_indepth, test_indepth_attripredCorr, test_show


def prototypeLearningStatus(model, unfreeze):
    for protoi in range(len(model.protodigis_list)):
        model.protodigis_list[protoi].requires_grad = unfreeze

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ProtoCaps")
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.02, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lam_recon', default=0.512, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--warmup', default=100, type=int,
                        help="Number of epochs before prototypes are fitted.")
    parser.add_argument('--push_step', default=10, type=int,
                        help="Prototypes are pushed every [push_step] epoch.")
    parser.add_argument('--split_number', default=0, type=int)
    parser.add_argument('--shareAttrLabels', default=1.0, type=float)
    parser.add_argument('--threeD', default=False, type=bool)
    parser.add_argument('--resize_shape', nargs='+', type=int, default=[32,32],  # for 3D experiments:[48, 48, 48]
                        help="Size of boxes cropped out of CT volumes as model input")
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    parser.add_argument('--epoch', type=int, help="Set the epoch of chosen model")
    # small out_dim_caps leads to different prototypes per attribute class
    parser.add_argument('--out_dim_caps', type=int, default=16, help="Set dimension of output capsule vectors.")
    parser.add_argument('--num_protos', type=int, default=2, help="Set number of prototypes per attribute class")
    args = parser.parse_args()
    print(args)

    if (not args.train and not args.test) or (args.train and args.test):
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True")

    train_loader, val_loader, test_loader = load_lidc(batch_size=args.batch_size,
                                                      resize_shape=args.resize_shape,
                                                      threeD=args.threeD,
                                                      splitnumber=args.split_number)

    numattributes = next(iter(train_loader))[2].shape[1]
    print("#attributes=#caps : " + str(numattributes))

    if args.test:
        if args.model_path == None:
            raise TypeError("Please specify the path to model by setting the parameter --model_path=\"path_to_model\"")
        if args.epoch == None:
            raise TypeError("Please specify the epoch of chosen model by setting the parameter --epoch=[int]")
        path = args.model_path
        model = torch.load(path)

        test_acc, test_attracc, test_dc = test_indepth(testmodel=model,
                                                       data_loader=test_loader,
                                                       epoch=args.epoch,
                                                       prototypefoldername=path.split("_")[0])
        print("PE_test_acc (Testing accuracy with use of prototypes): " + str(test_acc))
        print("PE_test_attr_acc: " + str(test_attracc))
        print("dc without exchange:" + str(test_dc))
        _, test_acc, te_attr_acc = test(testmodel=model, data_loader=test_loader, arguments=args)
        print('test acc = %.4f (within 1 score)' % test_acc)
        print("attr test acc = " + str(te_attr_acc))

        test_indepth_attripredCorr(testmodel=model,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   epoch=args.epoch,
                                   prototypefoldername=path.split("_")[0])

        # test_show(testmodel=model, data_loader=test_loader,
        #                            epoch=args.epoch,
        #                            prototypefoldername=path.split("_")[0])

    if args.train:
        model = ProtoCapsNet(input_size=[1, *args.resize_shape], numcaps=numattributes, routings=3,
                             out_dim_caps=args.out_dim_caps, activation_fn="sigmoid", threeD=args.threeD, numProtos=args.num_protos)
        model.cuda()
        print(model)

        opt_specs = [{'params': model.conv1.parameters(), 'lr': args.lr},
                     {'params': model.primarycaps.parameters(), 'lr': args.lr},
                     {'params': model.digitcaps.parameters(), 'lr': args.lr},
                     {'params': model.decoder.parameters(), 'lr': args.lr},
                     {'params': model.predOutLayers.parameters(), 'lr': args.lr},
                     {'params': model.relu.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer0.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer1.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer2.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer3.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer4.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer5.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer6.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer7.parameters(), 'lr': args.lr},
                     {'params': model.protodigis0, 'lr': 0.5},
                     {'params': model.protodigis1, 'lr': 0.5},
                     {'params': model.protodigis2, 'lr': 0.5},
                     {'params': model.protodigis3, 'lr': 0.5},
                     {'params': model.protodigis4, 'lr': 0.5},
                     {'params': model.protodigis5, 'lr': 0.5},
                     {'params': model.protodigis6, 'lr': 0.5},
                     {'params': model.protodigis7, 'lr': 0.5}]
        optimizer = torch.optim.Adam(opt_specs)

        print("training samples: " + str(len(train_loader.dataset)))
        print("val samples: " + str(len(val_loader.dataset)))
        print("test samples: " + str(len(test_loader.dataset)))
        train_samples_with_attrLabels_Loss = torch.randperm(len(train_loader.dataset))[
                                             :int(args.shareAttrLabels * len(train_loader.dataset))]
        print(str(len(train_samples_with_attrLabels_Loss)) + " samples are being considered of having attribute labels")

        best_val_acc = 0.0
        datestr = str(datetime.datetime.now())
        print("this run has datestr " + datestr)
        protosavedir = "./prototypes/" + str(datestr)
        os.mkdir(protosavedir)
        earlyStopping_counter = 1
        earlyStopping_max = 10  # push iterations
        for ep in range(args.epochs):
            if ep % args.push_step == 0:
                if ep >= args.warmup:
                    print("Pushing")
                    model, mindists_X, mindists_attr_sc = pushprotos(model_push=model, data_loader=train_loader,
                                                                     idx_with_attri=train_samples_with_attrLabels_Loss)
                    protosavedir = "./prototypes/" + str(datestr) + "/" + str(ep)
                    os.mkdir(protosavedir)
                    for cpsi in range(len(mindists_X)):
                        for proto_idx in range(mindists_X[cpsi].shape[0]):
                            for proto_idx2 in range(mindists_X[cpsi].shape[1]):
                                np.save(os.path.join(protosavedir + "/",
                                                     "cpslnr" + str(cpsi) + "_protonr" + str(
                                                         proto_idx) + "-" + str(proto_idx2) + "_gtattrcs" + str(
                                                         mindists_attr_sc[cpsi][proto_idx, proto_idx2])),
                                        mindists_X[cpsi][proto_idx, proto_idx2, 0])

                    valwProtoE_acc, valwProtoE_attracc, _ = test_indepth(testmodel=model,
                                                                         data_loader=val_loader,
                                                                         epoch=ep,
                                                                         prototypefoldername=datestr)
                    print("PE_val_acc: " + str(valwProtoE_acc))
                    print("PE_val_attracc: " + str(valwProtoE_attracc))

                    print("save model")
                    torch.save(model, str(datestr) + "_" + str(valwProtoE_acc) + '.pth')
                    if valwProtoE_acc > best_val_acc:
                        best_val_acc = valwProtoE_acc
                        earlyStopping_counter = 1
                    else:
                        earlyStopping_counter += 1
                        if earlyStopping_counter > earlyStopping_max:
                            sys.exit()

            if ep < args.warmup:
                prototypeLearningStatus(model, False)
            else:
                prototypeLearningStatus(model, True)
            print("Training")
            model, tr_acc, tr_attr_acc = train_model(
                model, train_loader, args, epoch=ep, optim=optimizer,
                idx_with_attri=train_samples_with_attrLabels_Loss)
            print('train acc = %.4f (within 1 score)' % tr_acc)

            print("Validation")
            val_loss, val_acc, _ = test(testmodel=model, data_loader=val_loader, arguments=args)
            print('val acc = %.4f (within 1 score), val loss = %.5f' % (
                val_acc, val_loss))

            print("Epoch " + str(ep) + ' ' + '-' * 70)

