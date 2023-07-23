import torchvision.models as models
import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self, resnet_type=50, pretrained=True):
        super(ResNet, self).__init__()
        if resnet_type == 18:
            self.model = models.resnet18(pretrained=pretrained)
        elif resnet_type == 34:
            self.model = models.resnet34(pretrained=pretrained)
        elif resnet_type == 50:
            self.model = models.resnet50(pretrained=pretrained)
        elif resnet_type == 101:
            self.model = models.resnet101(pretrained=pretrained)
        elif resnet_type == 152:
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError('resnet_type must be 18, 34, 50, 101 or 152')
        layers = list(self.model.children())
        self.init_layers = nn.Sequential(*layers)[:3]
        self.max_pool2d = layers[3]
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]
        self.avg_pool2d = layers[8]
        self.fc = layers[9]

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        inp = x.clone()
        # print('here:', inp.min(), inp.max())
        inp[:, 0:1, ...] = (x[:, 0:1, ...] - 0.485) / 0.229
        inp[:, 1:2, ...] = (x[:, 1:2, ...] - 0.456) / 0.224
        inp[:, 2:3, ...] = (x[:, 2:3, ...] - 0.406) / 0.225
        x0 = self.init_layers(inp)
        x1 = self.layer1(self.max_pool2d(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.avg_pool2d(x4)
        y = self.fc(torch.squeeze(x5))
        return y, [x0, x1, x2, x3]
