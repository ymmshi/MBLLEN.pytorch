import torch.nn as nn
import torch
from pytorch_msssim import MS_SSIM, SSIM
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg
from torchvision import transforms
from torch.nn import functional as F

class Loss(_Loss):
    def __init__(self, log):
        super(Loss, self).__init__()
        self.msssim = MS_SSIM(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, nonnegative_ssim=True)
        self.perceptual = PerceptualLoss()
        self.log = log

    def region(self, pred, label):
        gray = 0.30 * label[:,0,:,:] + 0.59 * label[:,1,:,:] + 0.11 * label[:,2,:,:]
        gray = gray.view(-1)
        value = -torch.topk(-gray, int(gray.shape[0] * 0.4))[0][0]
        weight = 1 * (label > value) + 4 * (label <= value)
        abs_diff = torch.abs(pred - label)
        return torch.mean(weight * abs_diff)

    def forward(self, x, y, mode):
        str_loss = 2 - self.msssim(x, y) - self.ssim(x, y)
        vgg_loss = self.perceptual(x, y)
        region_loss = self.region(x, y)
        self.log('%s_str_loss' % mode, str_loss)
        self.log('%s_vgg_loss' % mode, vgg_loss)
        self.log('%s_region_loss' % mode, region_loss)
        loss = str_loss + vgg_loss + region_loss
        return loss


class PerceptualLoss(_Loss):
    def __init__(self,):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg.vgg19(pretrained=True).features
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.vgg.eval()

    def vgg_forward(self, x):
        output = []
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name == '26':
                return x
    
    def preprocess(self, tensor):
        trsfrm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        res = trsfrm(tensor)
        return res       


    def forward(self, output, label):
        output = self.preprocess(output)
        label  = self.preprocess(label)
        feat_a = self.vgg_forward(output)
        feat_b = self.vgg_forward(label)

        return F.l1_loss(feat_a, feat_b)