import os
import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from gcd_loss import GuidedCorrespondenceLoss_forward


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def calc_mean_std(features):
    mean = []
    std = []
    for feature in features:
        assert len(feature.shape) == 4
        mean.append(torch.mean(feature, dim=(2, 3), keepdim=True))
        std.append(torch.std(feature, dim=(2, 3), unbiased=False, keepdim=True))
    return mean, std


def l2_norm(x):
    assert len(x.shape) == 4
    return torch.norm(x.view(x.shape[0], -1), dim=1).mean()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_tv_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_tv_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_tv_l1 = loss_tv_l1 * 255.0
    return loss_tv_l1, loss_tv_l2


def get_dm_loss(mean, std, target_mean, target_std):
    loss = 0
    for m, s, tm, ts in zip(mean, std, target_mean, target_std):
        loss += F.mse_loss(m, tm.detach()) + F.mse_loss(s, ts.detach())
    return 100.0 * loss


def load_img(path):
    img = Image.open(path).convert("RGB")
    # rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    # rgb_std = torch.tensor([0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    return trans(img).unsqueeze(0)


def load_label(path):
    img = Image.open(path).convert("L")
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    return trans(img).unsqueeze(0)

