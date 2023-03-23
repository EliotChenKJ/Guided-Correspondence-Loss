"""
Implement the texture synthesis pipeline(same resolution, without spatial tag) of the paper in PyTorch
"""
import os
import utils
import argparse
from time import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
from vgg_model import VGG19Model
from loss_fn import AugmentedGuidedCorrespondenceLoss_forward
from orientation_model import OrientationExtractor
from progression_refinement import progression_refinement


# ---------------------------------------------------------------------------------
# BASIC SETTINGS
# ---------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parser
parser = argparse.ArgumentParser(description='Texture generation')
parser.add_argument('--suffix', default='', help='output suffix')
parser.add_argument('--log_freq', default=100, type=int, help='logging frequency')
parser.add_argument('--base_iters', default=500, type=int, help='number of steps')
parser.add_argument('--finetune_iters', default=0, type=int, help='number of finetune steps')

parser.add_argument('--data_folder', default='', required=True, type=str, help='data folder')
parser.add_argument('--image_name', default='', required=True, type=str, help='image name')
parser.add_argument('--output_folder', default='./outputs', type=str, help='output folder')
parser.add_argument('--output_name', default='output.jpg', help='name of the output file')
parser.add_argument('--output_size', type=int, nargs='+', default=[512, 512], help='output size')
parser.add_argument('--size', default=0, type=int, help='resolution of the input texture (it will resize to this resolution)')
parser.add_argument('--scales', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1], help='multi-scale generation')
parser.add_argument('--num_augmentation', default=8, type=int, help='number of augmentation')
parser.add_argument('--use_flip', action='store_true', help='use flip augmentation')

parser.add_argument('--refer_prog_name', default='', type=str, help='input progression')
parser.add_argument('--trg_prog_name', default='', type=str, help='target progression')
parser.add_argument('--trg_orient_name', default='', type=str, help='target orientation')

parser.add_argument('--h', type=float, default=0.5, help='h lambdas')
parser.add_argument('--patch_size', type=int, default=7, help='patch size')
parser.add_argument('--feat_layers', type=int, nargs='+', default=[2, 4, 8], help='layers used in vgg')
parser.add_argument('--lambda_progression', type=float, default=0, help='progression lambdas')
parser.add_argument('--lambda_orientation', type=float, default=0, help='orientation lambdas')
parser.add_argument('--lambda_occurrence', type=float, default=0.05, help='occurrence lambdas')

parser.add_argument('--use_prog_reverse', action='store_true', help='reverse target progression')
parser.add_argument('--use_prog_dist_trans', action='store_true', help='distance transforming progression')
parser.add_argument('--use_prog_refine', action='store_true', help='refine target progression map')

args = parser.parse_args()
args.suffix = args.suffix + f'_resize({args.size})_ps({args.patch_size})_prog({args.lambda_progression})_orient({args.lambda_orientation})_occur({args.lambda_occurrence})_trgRefine({args.use_prog_refine})_trgReverse({args.use_prog_reverse})'
args.output_folder = args.output_folder + '/' + args.image_name + args.suffix
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

SIZE = args.size
SCALES = args.scales
INPUT_FILE = args.data_folder + '/source/' + args.image_name
OUTPUT_FILE = args.output_folder + '/' + args.output_name
FEAT_LAYERS = args.feat_layers
NB_ITER = args.base_iters + args.finetune_iters
ANGLES = [i * (360 / args.num_augmentation) for i in range(args.num_augmentation)]
USE_FLIP = args.use_flip

print(f'Launching texture synthesis from {INPUT_FILE} on size {SIZE} for {NB_ITER} steps. Output file: {OUTPUT_FILE}')


# ---------------------------------------------------------------------------------
# DATA - REFERENCE AND TARGET (IMAGE, PROGRESSION, ORIENTATION)
# ---------------------------------------------------------------------------------
def get_augmentation(tensor, use_flip=False, angles=None, generate_coordinate=False):
    # concat with coordinate
    if generate_coordinate:
        coordinate = utils.generate_coordinate(tensor)
        concat_tensor = torch.cat([tensor, coordinate], 1)
        meta_channel, coord_channel = tensor.shape[1], coordinate.shape[1]
    else:
        concat_tensor = tensor
        meta_channel, coord_channel = tensor.shape[1], 0

    # data augmentation
    meta_tensor_list = []
    coord_tensor_list = []

    if use_flip:
        tensor_list = [concat_tensor, concat_tensor.flip(dims=[-1]),
                       concat_tensor.flip(dims=[-2]), concat_tensor.flip(dims=[-1, -2])]
        for tensor in tensor_list:
            meta_tensor, coord_tensor = tensor.split([meta_channel, coord_channel], dim=1)
            meta_tensor = meta_tensor.clamp(0, 1)
            meta_tensor_list.append(meta_tensor)
            coord_tensor_list.append(coord_tensor)
    else:
        for angle in angles:
            rotated_tensor = rotate(concat_tensor.cpu(), angle, axes=(2, 3))
            w_new, h_new = utils.largest_rotated_rect(concat_tensor.shape[-1], concat_tensor.shape[-2], angle)
            rotated_tensor = utils.crop_around_center(rotated_tensor, w_new, h_new)
            rotated_tensor = torch.tensor(rotated_tensor).to(concat_tensor.device)
            meta_tensor, coord_tensor = rotated_tensor.split([meta_channel, coord_channel], dim=1)
            meta_tensor = meta_tensor.clamp(0, 1)
            meta_tensor_list.append(meta_tensor)
            coord_tensor_list.append(coord_tensor)

    return meta_tensor_list, coord_tensor_list

with torch.no_grad():
    # -----------------------------------------------
    # References (image, progression and orientation)
    # -----------------------------------------------
    # Reference image
    refer_texture = utils.decode_image(INPUT_FILE, size=SIZE).to(device)
    refer_textures, refer_coordinates = get_augmentation(refer_texture, use_flip=USE_FLIP, angles=ANGLES, generate_coordinate=True)

    for refer_idx, refer_tex in enumerate(refer_textures):
        utils.save_image(args.output_folder + f'/refer-texture-{refer_idx}.jpg', refer_tex)

    # Reference progression
    refer_progressions = None
    if args.lambda_progression > 0:
        progression_file = args.data_folder + '/source/' + args.refer_prog_name
        refer_progression = utils.decode_image(progression_file, size=SIZE, type='L').to(device)
        refer_progression = F.interpolate(refer_progression, size=refer_texture.shape[-2:], mode='bilinear')
        refer_progressions = get_augmentation(refer_progression, use_flip=USE_FLIP, angles=ANGLES)[0]

        for refer_idx, refer_prog in enumerate(refer_progressions):
            utils.visualize_progression_map(refer_prog, args.output_folder, f'refer-progression-{refer_idx}')

    # Reference orientation
    refer_orientations = None
    if args.lambda_orientation > 0:
        hogExtractor = OrientationExtractor().to(device)
        refer_orientations = [hogExtractor(refer_tex)[0] for refer_tex in refer_textures]

        for refer_idx, refer_orient in enumerate(refer_orientations):
            background_images = refer_progressions if args.lambda_progression > 0 else refer_textures
            utils.visualize_orientation_map(refer_orient, args.output_folder, f'refer-orientation-{refer_idx}',
                                            image=background_images[refer_idx])

    # -----------------------------------------------
    # Targets (image, progression, orientation)
    # -----------------------------------------------
    output_size = args.output_size

    # Target progression
    target_progression = None
    if args.lambda_progression > 0:
        progression_file = args.data_folder + '/target/' + args.trg_prog_name
        target_progression = utils.decode_image(progression_file, size=SIZE, type='L').to(device)
        target_progression = F.interpolate(target_progression, size=output_size, mode='bilinear')
        if args.use_prog_reverse: # progression reverse
            target_progression = 1 - target_progression
        if args.use_prog_refine: # progression refinement
            target_progression = progression_refinement(target_progression, refer_progression)

        utils.visualize_progression_map(target_progression, args.output_folder, 'target-progression')

    # Target orientation
    target_orientation = None
    if args.lambda_orientation > 0:
        target_orientation = np.load(args.data_folder + '/target/' + args.trg_orient_name)
        target_orientation = torch.from_numpy(target_orientation).type(torch.float32).to(device)[None]
        target_orientation = F.interpolate(target_orientation, size=output_size, mode='bilinear')
        target_orientation = target_orientation / (target_orientation.norm(dim=1, keepdim=True) + 1e-12)

        background_image = target_progression if args.lambda_progression > 0 else None
        utils.visualize_orientation_map(target_orientation, args.output_folder, 'target-orientation', image=background_image)

    if args.lambda_progression > 0: # initialize image
        target_texture = utils.initialize_from_progression(
            target_progression[0] * 255, refer_progression[0] * 255, refer_texture[0]
        )[None].to(device)
    else:
        target_grid = torch.rand([1, *output_size, 2]).to(device)
        target_grid = (target_grid - 0.5) * 2
        target_texture = F.grid_sample(refer_texture[:1], target_grid)

    utils.save_image(args.output_folder + '/target-init.jpg', target_texture)


# ---------------------------------------------------------------------------------
# FIT FUNCTION
# ---------------------------------------------------------------------------------
# Modules setup
extractor = VGG19Model(device, 3)  # customized_vgg_model in pytorch
loss_forward = AugmentedGuidedCorrespondenceLoss_forward(
    h=args.h,
    patch_size=args.patch_size,
    progression_weight=args.lambda_progression,
    orientation_weight=args.lambda_orientation,
    occurrence_weight=args.lambda_occurrence,
).to(device)

# Loss function
def get_loss(target_feats, refer_feats, progressions, orientations, refer_coordinates, layers=[]):
    loss = 0
    for layer in layers:
        loss += loss_forward(
            [target_feats[layer]], refer_feats[layer],
            progressions, orientations,
            refer_coordinates
        )
    return loss

# Optimization
target_sizes = [[int(output_size[0] * scale), int(output_size[1] * scale)] for scale in SCALES]

print('Start fitting...')
start = time()

for target_scale, target_size in zip(SCALES, target_sizes):
    target_texture = F.interpolate(target_texture, target_size).detach()
    optimizer = torch.optim.Adam([target_texture.requires_grad_()], lr=0.01)
    refer_feats = [[] for i in range(14)]
    for refer_idx, refer_tex in enumerate(refer_textures):
        feats = extractor(F.interpolate(refer_tex, scale_factor=target_scale))
        for feat_idx, feat in enumerate(feats):
            refer_feats[feat_idx].append(feat.detach())

    for i in range(NB_ITER):
        optimizer.zero_grad()
        target_texture.data.clamp_(0, 1)

        target_feats = extractor(target_texture)
        loss_ctx = get_loss(
            target_feats, refer_feats,
            [[target_progression], refer_progressions],
            [[target_orientation], refer_orientations],
            refer_coordinates,
            layers=FEAT_LAYERS
        )
        loss = loss_ctx
        loss.backward()
        optimizer.step()

        if i % args.log_freq == 0:
            target_texture.data.clamp_(0, 1)
            print(f'scale {target_scale} iter {i + 1} loss {loss.item()}')
            utils.save_image(args.output_folder + f'/output-scale{target_scale}-iter{i + 1}.jpg', target_texture)

target_texture.data.clamp_(0, 1)
utils.save_image(OUTPUT_FILE, target_texture)

end = time()
print('Time: {:} minutes'.format((end - start) / 60.0))

