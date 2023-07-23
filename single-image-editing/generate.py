import utils
import torch
import torch.nn as nn
from tqdm import trange


from resnet import ResNet
from vgg_model import VGG19Model
import torch.nn.functional as F
from gcd_loss import GuidedCorrespondenceLoss_forward
from torchvision.utils import save_image

loss_forward = GuidedCorrespondenceLoss_forward(
    h=0.5,
    patch_size=7,
    progression_weight=10.0,
    occurrence_weight=0.1,
).cuda()


# Loss function
def get_gcd_loss(syn_feats, target_feats, layers, label=None):
    loss = 0
    for i in layers:
        loss += loss_forward(syn_feats[i], target_feats[i].detach(), label)
    return loss


def generate(init_img, reference_img, device, name, use_layout=False):

    lr = 0.01
    lambda_tv = 1e-3
    lambda_l2 = 1e-5
    lambda_dm = 5.0
    lambda_ce = 1.0
    lambda_gcd = 5.0
    iterations_per_scale = 500
    h, w = init_img.shape[2:]
    base_layers = [2, 4, 8]
    fineTune_layers = [0, 2, 4, 8]
    scales = [0.25, 0.5, 0.75, 1.0]
    sizes = [[int(h * s), int(w * s)] for s in scales]

    syn_img = init_img

    label = None
    layers = None
    if use_layout:
        reference_label = utils.load_label(f"./data/label/reference/{name}.png").to(device)
        target_label = utils.load_label(f"./data/label/target/{name}.png").to(device)
        label = target_label, reference_label

    CELoss = nn.CrossEntropyLoss()
    resnet_model = ResNet().to(device)
    vgg_model = VGG19Model(device, 3)
    resnet_model.eval().requires_grad_(False)
    vgg_model.eval().requires_grad_(False)

    for scale, size in zip(scales, sizes):
        syn_img = F.interpolate(syn_img, size=size).detach()
        target_img = F.interpolate(reference_img, size=size).detach()

        optim = torch.optim.Adam([syn_img.requires_grad_()], lr=lr)
        target_vgg_features = vgg_model(target_img)
        target_resnet_class, target_resnet_features = resnet_model(target_img)
        target_resnet_class = target_resnet_class.unsqueeze(0)
        target_resnet_class = torch.max(target_resnet_class, dim=1)[1]
        target_mean, target_std = utils.calc_mean_std(target_resnet_features)
        print("Target class: ", target_resnet_class)
        with trange(iterations_per_scale) as t:
            t.set_description(f"name:{name} scale:{scale}") 
            for iteration in t:
                optim.zero_grad()
                syn_vgg_features = vgg_model(syn_img)
                syn_resnet_class, syn_resnet_features = resnet_model(syn_img)
                syn_resnet_class = syn_resnet_class.unsqueeze(0)
                mean, std = utils.calc_mean_std(syn_resnet_features)

                loss_ce = CELoss(syn_resnet_class, target_resnet_class.detach())
                loss_tv_l1, loss_tv_l2 = utils.get_image_prior_losses(syn_img)
                loss_l2 = utils.l2_norm(syn_img)
                loss_dm = utils.get_dm_loss(mean, std, target_mean, target_std)
                if iteration < 300:
                    layers = base_layers
                else:
                    layers = fineTune_layers

                loss_gcd = get_gcd_loss(syn_vgg_features, target_vgg_features, layers, label)
                loss = lambda_gcd * loss_gcd + \
                       lambda_tv * (loss_tv_l1 + loss_tv_l2) + \
                       lambda_l2 * loss_l2 + \
                       lambda_dm * loss_dm + \
                       lambda_ce * loss_ce

                loss.backward()
                optim.step()
                syn_img.data.clamp_(0, 1)
                t.set_postfix(loss=loss.item(), gcd_loss=loss_gcd.item())  

                if (iteration+1) % 100 == 0:
                    save_image(syn_img, f'outputs/{name}/scale_{scale}_{iteration+1}.jpg')
        save_image(syn_img, f'outputs/{name}.jpg')
