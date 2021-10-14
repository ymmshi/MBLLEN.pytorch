import torch.nn as nn
import torch


class MBLLEN(nn.Module):
    def __init__(self, em_channel=8, fem_channel=32, block_num=9):
        super(MBLLEN, self).__init__()
        self.block_num = block_num
        self.FEM = nn.Conv2d(3, fem_channel, (3, 3), padding=1)
        self.EM = EM(em_channel)

        self.activation = nn.ReLU(inplace=True)

        blocks = []
        for _ in range(self.block_num):
            blocks += [nn.Conv2d(fem_channel, fem_channel, (3, 3), padding=1)]
            blocks += [EM(em_channel)]
        self.blocks = nn.Sequential(*blocks)

        self.FM = nn.Conv2d(3 * (self.block_num + 1), 3, (1, 1))

    def forward(self, x):
        x = self.FEM(x)
        x = self.activation(x)
        em_features = []
        em_features.append(self.EM(x))
        for i in range(self.block_num):
            x = self.blocks[2 * i](x)
            em_features.append(self.blocks[2 * i + 1](x))
        em_features = torch.cat(em_features, dim=1)
        out = self.FM(em_features)
        return torch.clamp(out, 0, 1)


class EM(nn.Module):
    def __init__(self, channel=8):
        super(EM, self).__init__()
        model = []
        model += [nn.Conv2d(32, 8, (3, 3), padding=1)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d(channel, channel, (5, 5), padding=2)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d(channel, channel*2, (5, 5), padding=2)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Conv2d(channel*2, channel*4, (5, 5), padding=2)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(channel*4, channel*2, (5, 5), padding=2)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(channel*2, channel, (5, 5), padding=2)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.ConvTranspose2d(channel, 3, (5, 5), padding=2)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)