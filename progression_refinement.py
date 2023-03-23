import torch
import torch.nn.functional as F


def progression_refinement(target_progression, refer_progression, num_levels=7):
    assert target_progression.shape[0] == 1 and target_progression.shape[1] == 1, 'progression refinement only works in num_batches == 1 && num_channels == 1'

    # construct laplacian pyramid
    target_laplacian_pyramid = construct_laplacian_pyramid(target_progression, num_levels=num_levels)
    refer_laplacian_pyramid = construct_laplacian_pyramid(refer_progression, num_levels=num_levels)

    # add noise per level
    output_laplacian_pyramid = match_and_add_noise(target_laplacian_pyramid, refer_laplacian_pyramid)

    # reconstruction
    output_progression = reconstruct_laplacian_pyramid(output_laplacian_pyramid)
    output_progression = histogram_matching(output_progression, refer_progression)

    return normalization(output_progression, min=refer_progression.min(), max=refer_progression.max())


def construct_laplacian_pyramid(image, num_levels=7):
    gauss_pyramid = [image]

    # construct guass pyramid
    for i in range(1, num_levels):
        downsampled_image = F.interpolate(gauss_pyramid[-1], scale_factor=0.5, mode='bilinear')
        gauss_pyramid.append(downsampled_image)

    # construct laplacian pyramid
    laplacian_pyramid = [None] * num_levels
    laplacian_pyramid[-1] = gauss_pyramid[-1]
    for i in range(num_levels - 2, -1, -1):
        upsampled_image = F.interpolate(gauss_pyramid[i+1], size=gauss_pyramid[i].shape[-2:], mode='bilinear')
        laplacian_pyramid[i] = gauss_pyramid[i] - upsampled_image

    return laplacian_pyramid


def match_and_add_noise(target_pyramid, refer_pyramid):
    num_levels = len(target_pyramid)
    output_pyramid = [None] * num_levels

    # histogram matching in last level
    output_pyramid[-1] = histogram_matching(target_pyramid[-1], refer_pyramid[-1])

    # add random noise in laplacian pyramid
    for i in range(num_levels - 1):
        current_target = target_pyramid[i]
        current_refer = refer_pyramid[i]
        num_batches, num_channels, target_height, target_width = current_target.shape

        # random noise(same size with target)
        noise = torch.randn(num_batches, num_channels, target_height // 2, target_width // 2).to(current_target.device)
        noise = F.interpolate(noise, size=[target_height, target_width], mode='bilinear')

        # adjust noise range
        noise_var = noise.var(dim=[1, 2, 3])
        refer_var = current_refer.var(dim=[1, 2, 3])
        deviation = refer_var / noise_var * noise

        output_pyramid[i] = target_pyramid[i] + deviation

    return output_pyramid


def reconstruct_laplacian_pyramid(laplacian_pyramid):
    num_levels = len(laplacian_pyramid)

    # reconstructing image from laplacian pyramid
    current_image = laplacian_pyramid[-1]
    for i in range(num_levels - 2, -1, -1):
        upsampled_current_image = F.interpolate(current_image, size=laplacian_pyramid[i].shape[-2:], mode='bilinear')
        current_image = upsampled_current_image + laplacian_pyramid[i]
    
    reconstructed_image = current_image
    
    return reconstructed_image


def normalization(tensor, min=0, max=1):
    _min = tensor.view(1, -1).min(dim=1)[0]
    _max = tensor.view(1, -1).max(dim=1)[0]

    return (tensor - _min) / (_max - _min) * (max - min) + min


def histogram_matching(source, template):
    """Adjust the pixel values of an image to match its histogram towards a target image.
    `Histogram matching <https://en.wikipedia.org/wiki/Histogram_matching>`_ is the transformation
    of an image so that its histogram matches a specified histogram. In this implementation, the
    histogram is computed over the flattened image array. Code referred to
    `here <https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x>`_.
    Args:
        source: Image to transform.
        template: Template image. It can have different dimensions to source.
    Returns:
        The transformed output image as the same shape as the source image.
    Note:
        This function does not matches histograms element-wisely if input a batched tensor.
    """

    oldshape = source.shape

    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts.
    s_values, bin_idx, s_counts = torch.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = torch.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)

    s_quantiles = torch.cumsum(s_counts, dim=0)
    s_quantiles = s_quantiles / s_quantiles[-1]
    t_quantiles = torch.cumsum(t_counts, dim=0)
    t_quantiles = t_quantiles / t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Interpolate ``x`` tensor according to ``xp`` and ``fp`` as in ``np.interp``.
    This implementation cannot reproduce numpy results identically, but reasonable.
    Code referred to `here <https://github.com/pytorch/pytorch/issues/1552#issuecomment-926972915>`_.
    Args:
        x: the input tensor that needs to be interpolated.
        xp: the x-coordinates of the referred data points.
        fp: the y-coordinates of the referred data points, same length as ``xp``.
    Returns:
        The interpolated values, same shape as ``x``.
    """
    if x.dim() != xp.dim() != fp.dim() != 1:
        raise ValueError(
            f"Required 1D vector across ``x``, ``xp``, ``fp``. Got {x.dim()}, {xp.dim()}, {fp.dim()}."
        )

    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    locs = torch.searchsorted(xp, x)
    locs = locs.clip(1, len(xp) - 1) - 1
    return slopes[locs] * (x - xp[locs]) + fp[locs]


# -------------------------------
# import PIL.Image as Image
# import torchvision.transforms as tf
# import matplotlib.pyplot as plt
# import numpy as np
#
# def show_image(tensor):
#     np_tensor = (tensor[0] * 255.).repeat(3,1,1).permute(1, 2, 0).numpy().astype(np.uint8)
#     image = Image.fromarray(np_tensor)
#     plt.imshow(image)
#     plt.show()
#
#
# source = Image.open('source.png').convert('L')
# source_tensor = (tf.ToTensor()(source)[None, :, :, :] - 0.5) * 2
#
# target = Image.open('target.png').convert('L')
# target_tensor = (tf.ToTensor()(target)[None, :, :, :] - 0.5) * 2
#
# prog = progression_refinement(target_progression=target_tensor, refer_progression=source_tensor)
# print()
