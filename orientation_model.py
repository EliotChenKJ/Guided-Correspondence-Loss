import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import math
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

class OrientationExtractor(nn.Module):
    def __init__(self, nBins=9, cellSize=16, minAngle=0, maxAngle=180, epsilon=1e-9):
        super(OrientationExtractor, self).__init__()
        self.nBins = nBins
        self.cellSize = cellSize
        self.minAngle = minAngle
        self.maxAngle = maxAngle
        self.epsilon = epsilon

        grayFilter = torch.FloatTensor([0.299, 0.587, 0.114])
        self.register_buffer('grayFilter', grayFilter[None, :, None, None])

        gradientFilter = torch.FloatTensor([-1, 0, 1])
        self.register_buffer('gradXFilter', gradientFilter[None, None, None, :])
        self.register_buffer('gradYFilter', gradientFilter[None, None, :, None].clone())

        binFilters = torch.linspace(1, nBins, nBins)
        binInterval = maxAngle / nBins
        binCenters = torch.linspace(binInterval / 2, maxAngle - binInterval / 2, nBins)
        self.register_buffer('binCenters', binCenters[None, :, None, None])
        self.register_buffer('binFilters', binFilters[None, :, None, None])

        self.register_buffer('aggregateFilter', self.getAggregateFilter())

    def getAggregateFilter(self):
        # aggregate filter related to distance
        cellSize = self.cellSize
        aggregateFilter = torch.zeros(2 * cellSize + 1, 2 * cellSize + 1)
        filterCenter = cellSize
        maxLength = filterCenter * 2 ** 0.5
        minLength = 0
        for i in range(2 * cellSize + 1):
            for j in range(2 * cellSize + 1):
                aggregateFilter[i, j] = ((i - filterCenter) ** 2 + (j - filterCenter) ** 2) ** 0.5
        aggregateFilter = 1 - (aggregateFilter - minLength) / (maxLength - minLength)

        return aggregateFilter[None, None, :, :].expand(self.nBins, 1, -1, -1)

    def getHog(self, input):
        # calculate gradient
        gradientX = F.conv2d(input, self.gradXFilter, padding=[0, 1])
        gradientY = F.conv2d(input, self.gradYFilter, padding=[1, 0])
        intensity = torch.sqrt(gradientX ** 2 + gradientY ** 2)

        # binning
        orientation = (torch.atan2(gradientY, gradientX) + math.pi) % math.pi / math.pi * self.maxAngle
        orientationBins = torch.clamp(
            1 - torch.abs(orientation - self.binCenters) * self.nBins / self.maxAngle, 0, 1)

        # hog extraction
        weightedOrientationBins = orientationBins * intensity
        paddBins = F.pad(weightedOrientationBins, [self.cellSize] * 4, mode='reflect')
        hog = F.conv2d(paddBins, self.aggregateFilter, stride=self.cellSize, groups=self.nBins)
        hog = hog / torch.norm(hog, dim=1).view(hog.shape[0], -1).max(-1)[0][:, None, None, None]

        return hog

    def getDominantOrientation(self, hog):
        # dominant orientation
        numBatch, _, hogHeight, hogWidth = hog.shape
        orientMatrix = torch.stack([
            torch.sin(self.binCenters / self.maxAngle * math.pi) * hog,
            -torch.cos(self.binCenters / self.maxAngle * math.pi) * hog
        ], dim=1)
        orientMatrix = torch.cat([orientMatrix, -orientMatrix], 2).permute(0, 3, 4, 1, 2).reshape(-1, 2, self.nBins * 2)
        _, s, v = torch.svd(torch.bmm(orientMatrix, orientMatrix.permute(0, 2, 1)))
        v = v[:, :, 0]
        domiantOrientation = v.view(numBatch, hogHeight, hogWidth, 2).permute(0, 3, 1, 2)

        return domiantOrientation

    def getResizedOrientation(self, orient, target_size):
        orient_image = torch.cat([orient, torch.ones_like(orient[:, :1]) * -1], 1).cpu()

        output_orients = []
        for index in range(orient.shape[0]):
            current_image = (((orient_image + 1) / 2) * 255)[index].permute(1, 2, 0).type(torch.uint8)
            current_image = Image.fromarray(current_image.numpy())
            current_image = tf.functional.resize(current_image, target_size,
                                                interpolation=tf.InterpolationMode.LANCZOS)
            output_orient = torch.tensor(np.array(current_image), device=orient.device)
            output_orient = (output_orient[:, :, :2] / 255 - 0.5) * 2
            output_orient = output_orient.permute(2, 0, 1)[None]
            output_orient = output_orient / output_orient.norm(dim=1)
            output_orients.append(output_orient)

        return torch.cat(output_orients, 0)

    def forward(self, input, return_hog=False, return_visual=False):
        input_size = input.shape[-2:]

        # preprocess
        assert len(input.shape) == 4, 'type of data is wrong'
        if input.shape[1] == 3:
            input = F.conv2d(input, self.grayFilter)

        hog = self.getHog(input)
        orientation = self.getDominantOrientation(hog)
        orientation = self.getResizedOrientation(orientation, input_size)

        return_list = [orientation]
        if return_hog:
            return_list.append(hog)
        if return_visual:
            return_list.append(self.getOrientationVisualization(input, orientation))

        return return_list

    def getOrientationVisualization(self, image, orientation):
        numBatch, _, imageHeight, imageWidth = image.shape
        numBatch, _, orientationHeight, orientationWidth = orientation.shape

        # draw visualization each image
        visualizationList = []
        for i in range(numBatch):
            # stride = 1 -----------------------------------
            orient = (torch.cat([orientation[i], torch.ones_like(orientation[i][:1]) * -1], 0) + 1) / 2 * 255.
            orient = orient.type(torch.uint8).permute(1, 2, 0).cpu().numpy()
            visualizeImage = Image.fromarray(orient)
            visualizationList.append(visualizeImage)

        return visualizationList


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cuda = True
    path = 'example.jpg'

    # reference image
    image = Image.open(path).convert('L')
    im = tf.ToTensor()(image)
    x = im[None]
    if cuda:
        x = x.cuda()

    # target image
    target = torch.rand_like(x) * 255
    target.requires_grad_(True)
    if cuda:
        target = target.cuda()

    # module
    extractor = OrientationExtractor()
    if cuda:
        extractor = extractor.cuda()

    orientation, hog, vis_list = extractor(x, visualize_orientation=True)

    plt.imshow(vis_list['orientation'][0])
    plt.show()