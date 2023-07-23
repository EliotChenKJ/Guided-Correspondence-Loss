"""
Implement the customization of VGG-19 in pytorch, may be inconsistent to the official tf2 version
"""
import torch
import torch.nn as nn
import copy
import torchvision.models as models


def get_model(device):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            layer.padding_mode = 'reflect'
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
    return model


class VGG19Model(torch.nn.Module):
    def __init__(self, device, input_channel=3):
        super(VGG19Model, self).__init__()
        self.device = device
        # self.cnn = cnn
        model = get_model(device)
        self.input_channel = input_channel

        # make name consistent
        # print('start')
        self.block1_conv1 = model[0:2]  # [0]
        # print(self.block1_conv1)
        self.block1_conv2 = model[2:4]
        # print(self.block1_conv2)

        self.block2_conv1 = model[4:7]  # [2]
        # print(self.block2_conv1)
        self.block2_conv2 = model[7:9]
        # print(self.block2_conv2)

        self.block3_conv1 = model[9:12]  # [4]
        # print(self.block3_conv1)
        self.block3_conv2 = model[12:14]
        # print(self.block3_conv2)
        self.block3_conv3 = model[14:16]
        # print(self.block3_conv3)
        self.block3_conv4 = model[16:18]
        # print(self.block3_conv4)

        self.block4_conv1 = model[18:21]  # [8]
        # print(self.block4_conv1)
        self.block4_conv2 = model[21:23]
        # print(self.block4_conv2)
        self.block4_conv3 = model[23:25]
        # print(self.block4_conv3)
        self.block4_conv4 = model[25:27]
        # print(self.block4_conv4)

        self.block5_conv1 = model[27:30]  # [12]
        # print(self.block5_conv1)
        self.block5_conv2 = model[30:32]
        # print(self.block5_conv2)

    def forward(self, x):
        # imagenet
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        inp = x.clone()
        # print('here:', inp.min(), inp.max())
        inp[:, 0:1, ...] = (x[:, 0:1, ...] - 0.485) / 0.229
        inp[:, 1:2, ...] = (x[:, 1:2, ...] - 0.456) / 0.224
        inp[:, 2:3, ...] = (x[:, 2:3, ...] - 0.406) / 0.225

        outputs = []
        x = self.block1_conv1(inp)
        outputs.append(x)
        # print(x.shape)
        x = self.block1_conv2(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block2_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block2_conv2(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block3_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block3_conv2(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block3_conv3(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block3_conv4(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block4_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block4_conv2(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block4_conv3(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block4_conv4(x)
        outputs.append(x)
        # print(x.shape)

        x = self.block5_conv1(x)
        outputs.append(x)
        # print(x.shape)
        x = self.block5_conv2(x)
        outputs.append(x)
        # print(x.shape)

        return outputs
