"""
Proto-Caps model

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import torch
from torch import nn

from capsulelayers import DenseCapsule, PrimaryCapsule


class ProtoCapsNet(nn.Module):
    def __init__(self, input_size, numcaps, routings, out_dim_caps, activation_fn, threeD, numProtos):
        super(ProtoCapsNet, self).__init__()
        self.input_size = input_size
        self.numcaps = numcaps
        self.routings = routings
        self.out_dim_caps = out_dim_caps
        self.numclasses = 5
        self.threeD = threeD
        self.numProtos = numProtos

        # Layer 1: Just a conventional Conv2D layer
        if self.threeD:
            self.conv1 = nn.Conv3d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, threeD=self.threeD, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        if self.threeD:
            self.digitcaps = DenseCapsule(in_num_caps=131072, in_dim_caps=8,
                                          out_num_caps=numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                          activation_fn=activation_fn)
        else:
            self.digitcaps = DenseCapsule(in_num_caps=32 * 8 * 8, in_dim_caps=8,
                                          out_num_caps=numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                          activation_fn=activation_fn)
        # Decoder network.
        if self.threeD:
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(numcaps * out_dim_caps, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, input_size[0] * input_size[1] * input_size[2] * input_size[3]),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(numcaps * out_dim_caps, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
                nn.Sigmoid()
            )

        # Prediction layers
        self.predOutLayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=numcaps * out_dim_caps, out_features=self.numclasses),
            nn.Softmax(dim=-1)
        )

        self.relu = nn.ReLU()

        # Attribute layers
        self.attrOutLayer0 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer1 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer2 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer3 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer4 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer5 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer6 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer7 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )

        # Prototype vectors
        self.protodigis0 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis1 = nn.Parameter(torch.rand((4, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis2 = nn.Parameter(torch.rand((6, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis3 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis4 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis5 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis6 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis7 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                self.protodigis4, self.protodigis5, self.protodigis6, self.protodigis7]

    def forwardCapsule(self, x_ex):
        capsout0 = self.attrOutLayer0(x_ex[:, 0, :])
        capsout1 = self.attrOutLayer1(x_ex[:, 1, :])
        capsout2 = self.attrOutLayer2(x_ex[:, 2, :])
        capsout3 = self.attrOutLayer3(x_ex[:, 3, :])
        capsout4 = self.attrOutLayer4(x_ex[:, 4, :])
        capsout5 = self.attrOutLayer5(x_ex[:, 5, :])
        capsout6 = self.attrOutLayer6(x_ex[:, 6, :])
        capsout7 = self.attrOutLayer7(x_ex[:, 7, :])
        pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7), dim=1)
        reconstruction = self.decoder(x_ex)
        pred = self.predOutLayers(x_ex)

        return pred, pred_attr, reconstruction.view(-1, *self.input_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)

        # attribute out
        capsout0 = self.attrOutLayer0(x[:, 0, :])
        capsout1 = self.attrOutLayer1(x[:, 1, :])
        capsout2 = self.attrOutLayer2(x[:, 2, :])
        capsout3 = self.attrOutLayer3(x[:, 3, :])
        capsout4 = self.attrOutLayer4(x[:, 4, :])
        capsout5 = self.attrOutLayer5(x[:, 5, :])
        capsout6 = self.attrOutLayer6(x[:, 6, :])
        capsout7 = self.attrOutLayer7(x[:, 7, :])
        pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7), dim=1)

        # reconstruction
        reconstruction = self.decoder(x)

        # prediction out
        pred = self.predOutLayers(x)

        dists_to_protos = self.getDistance(x)

        return pred, pred_attr, reconstruction.view(-1, *self.input_size), dists_to_protos

    def getDistance(self, x):
        """
        Capsule wise calculation of distance to closest prototype vector
        :param x: vectors to calculate distance to
        :return: distances to closest protoype vector
        """

        xreshaped = torch.unsqueeze(x, dim=1)
        xreshaped = torch.unsqueeze(xreshaped, dim=1)
        protoreshaped0 = torch.unsqueeze(self.protodigis0, dim=0)
        protoreshaped1 = torch.unsqueeze(self.protodigis1, dim=0)
        protoreshaped2 = torch.unsqueeze(self.protodigis2, dim=0)
        protoreshaped3 = torch.unsqueeze(self.protodigis3, dim=0)
        protoreshaped4 = torch.unsqueeze(self.protodigis4, dim=0)
        protoreshaped5 = torch.unsqueeze(self.protodigis5, dim=0)
        protoreshaped6 = torch.unsqueeze(self.protodigis6, dim=0)
        protoreshaped7 = torch.unsqueeze(self.protodigis7, dim=0)

        dists_0 = (xreshaped[:, :, :, 0, :] - protoreshaped0).pow(2).sum(-1).sqrt()

        dists_1 = (xreshaped[:, :, :, 1, :] - protoreshaped1).pow(2).sum(-1).sqrt()

        dists_2 = (xreshaped[:, :, :, 2, :] - protoreshaped2).pow(2).sum(-1).sqrt()

        dists_3 = (xreshaped[:, :, :, 3, :] - protoreshaped3).pow(2).sum(-1).sqrt()

        dists_4 = (xreshaped[:, :, :, 4, :] - protoreshaped4).pow(2).sum(-1).sqrt()

        dists_5 = (xreshaped[:, :, :, 5, :] - protoreshaped5).pow(2).sum(-1).sqrt()

        dists_6 = (xreshaped[:, :, :, 6, :] - protoreshaped6).pow(2).sum(-1).sqrt()

        dists_7 = (xreshaped[:, :, :, 7, :] - protoreshaped7).pow(2).sum(-1).sqrt()

        dists_to_protos = [dists_0, dists_1, dists_2, dists_3, dists_4, dists_5, dists_6, dists_7]

        return dists_to_protos
