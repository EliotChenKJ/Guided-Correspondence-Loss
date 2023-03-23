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
from loss_fn import GuidedCorrespondenceLoss_forward
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
parser.add_argument('--base_iters', default=300, type=int, help='number of steps')
parser.add_argument('--finetune_iters', default=200, type=int, help='number of finetune steps')

parser.add_argument('--data_folder', default='', required=True, type=str, help='data folder')
parser.add_argument('--image_name', default='', required=True, type=str, help='image name')
parser.add_argument('--output_folder', default='./outputs', type=str, help='output folder')
parser.add_argument('--output_name', default='output.jpg', help='name of the output file')
parser.add_argument('--output_size', type=int, nargs='+', default=[512, 512], help='output size')
parser.add_argument('--size', default=0, type=int, help='resolution of the input texture (it will resize to this resolution)')
parser.add_argument('--scales', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1], help='multi-scale generation')
parser.add_argument('--base_layers', type=int, nargs='+', default=[2, 4, 8], help='layers used in vgg')
parser.add_argument('--finetune_layers', type=int, nargs='+', default=[0, 2, 4, 8], help='layers used in vgg')

parser.add_argument('--refer_prog_name', default='', type=str, help='input progression')
parser.add_argument('--trg_prog_name', default='', type=str, help='target progression')
parser.add_argument('--trg_orient_name', default='', type=str, help='target orientation')
parser.add_argument('--use_prog_reverse', action='store_true', help='reverse target progression')
parser.add_argument('--use_prog_dist_trans', action='store_true', help='distance transforming progression')
parser.add_argument('--use_prog_refine', action='store_true', help='refine target progression map')

parser.add_argument('--h', type=float, default=0.5, help='h lambdas')
parser.add_argument('--patch_size', type=int, default=7, help='patch size')
parser.add_argument('--lambda_progression', type=float, default=0, help='progression lambdas')
parser.add_argument('--lambda_orientation', type=float, default=0, help='orientation lambdas')
parser.add_argument('--lambda_occurrence', type=float, default=0.05, help='occurrence lambdas')

args = parser.parse_args()
args.suffix = args.suffix + f'_resize({args.size})_ps({args.patch_size})_prog({args.lambda_progression})_orient({args.lambda_orientation})_occur({args.lambda_occurrence})_trgRefine({args.use_prog_refine})_trgReverse({args.use_prog_reverse})'
args.output_folder = args.output_folder + '/' + args.image_name + args.suffix
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

SIZE = args.size
OUTPUT_FILE = args.output_folder + '/' + args.output_name
INPUT_FILE = args.data_folder + '/source/' + args.image_name
NB_ITER = args.base_iters + args.finetune_iters

base_layers = args.base_layers
fineTune_layers = args.finetune_layers
scales = args.scales

print(f'Launching texture synthesis from {INPUT_FILE} on size {SIZE} for {NB_ITER} steps. Output file: {OUTPUT_FILE}')


# ---------------------------------------------------------------------------------
# DATA - REFERENCE AND TARGET (IMAGE, PROGRESSION, ORIENTATION)
# ---------------------------------------------------------------------------------
with torch.no_grad():
    # -----------------------------------------------
    # References (image, progression and orientation)
    # -----------------------------------------------
    refer_texture = utils.decode_image(INPUT_FILE, size=SIZE).to(device)
    utils.save_image(args.output_folder + '/refer-texture.jpg', refer_texture)

    # Reference progression
    refer_progression = None
    if args.lambda_progression > 0:
        progression_file = args.data_folder + '/source/' + args.refer_prog_name
        refer_progression = utils.decode_image(progression_file, size=SIZE, type='L').to(device)
        refer_progression = F.interpolate(refer_progression, size=refer_texture.shape[-2:])
        if args.use_prog_dist_trans:
            refer_progression = utils.distance_transform(refer_progression)
        utils.visualize_progression_map(refer_progression, args.output_folder, 'refer-progression')

    # Reference orientation
    refer_orientation = None
    if args.lambda_orientation > 0:
        hogExtractor = OrientationExtractor().to(device)
        refer_orientation = hogExtractor(refer_texture)[0]
        utils.visualize_orientation_map(refer_orientation, args.output_folder, 'refer-orientation', image=refer_texture)

    # -----------------------------------------------
    # Targets (image, progression, orientation)
    # -----------------------------------------------
    output_size = args.output_size

    # Target progression
    target_progression = None
    if args.lambda_progression > 0:
        progression_file = args.data_folder + '/target/' + args.trg_prog_name
        target_progression = utils.decode_image(progression_file, size=SIZE, type='L').to(device)
        target_progression = F.interpolate(target_progression, size=output_size)
        if args.use_prog_reverse:  # progression reverse
            target_progression = 1 - target_progression
        if args.use_prog_dist_trans: # progression transform
            target_progression = utils.distance_transform(target_progression)
        if args.use_prog_refine: # progression refinement
            target_progression = progression_refinement(target_progression, refer_progression)
        utils.visualize_progression_map(target_progression, args.output_folder, 'target-progression')

    # Target orientation
    target_orientation = None
    if args.lambda_orientation > 0:
        # target  --------------------
        target_orientation = np.load(args.trg_orient_name)
        target_orientation = torch.from_numpy(target_orientation).type(torch.float32).to(device)[None]
        target_orientation = F.interpolate(target_orientation, size=output_size, mode='bilinear')
        target_orientation = target_orientation / target_orientation.norm(dim=1, keepdim=True)
        # one direction --------------
        # target_orientation = F.interpolate(refer_orientation, size=output_size[-2:])
        # target_orientation = torch.ones_like(target_orientation) * \
        #                      torch.tensor([0, 1])[None, :, None, None].to(refer_orientation.device)
        # target_orientation = torch.ones_like(target_orientation) * \
        #                      torch.tensor([1, 0])[None, :, None, None].to(refer_orientation.device)
        # target_orientation = torch.ones_like(target_orientation) * \
        #                      torch.tensor([0.7, 0.7])[None, :, None, None].to(refer_orientation.device)
        # target_orientation = torch.ones_like(target_orientation) * \
        #                      torch.tensor([-0.7, 0.7])[None, :, None, None].to(refer_orientation.device)
        # ----------------------------
        utils.visualize_orientation_map(target_orientation, args.output_folder, 'target-orientation')

    if args.lambda_progression > 0: # initialize image
        target_texture = utils.initialize_from_progression(
            target_progression[0] * 255, refer_progression[0] * 255, refer_texture[0]
        )[None].to(device)
    else:
        target_texture = torch.rand([1, 3, *output_size]).to(device)

    utils.save_image(args.output_folder + '/target-init.jpg', target_texture)


# ---------------------------------------------------------------------------------
# FIT FUNCTION
# ---------------------------------------------------------------------------------
# Modules setup
extractor = VGG19Model(device, 3)  # customized_vgg_model in pytorch
loss_forward = GuidedCorrespondenceLoss_forward(
    h=args.h,
    patch_size=args.patch_size,
    progression_weight=args.lambda_progression,
    orientation_weight=args.lambda_orientation,
    occurrence_weight=args.lambda_occurrence,
).to(device)

# Loss function
def get_loss(target_feats, refer_feats, progressions, orientations, layers=[]):
    loss = 0
    for layer in layers:
        loss += loss_forward(
            target_feats[layer], refer_feats[layer],
            progressions, orientations,
        )
    return loss

# Optimization
target_sizes = [[int(output_size[0] * scale), int(output_size[1] * scale)] for scale in scales]

print('Start fitting...')
start = time()

for target_scale, target_size in zip(scales, target_sizes):
    target_texture = F.interpolate(target_texture, target_size).detach()
    optimizer = torch.optim.Adam([target_texture.requires_grad_()], lr=0.01)
    refer_feats = [feat.detach() for feat in extractor(F.interpolate(refer_texture, scale_factor=target_scale))]

    for iter in range(NB_ITER):
        optimizer.zero_grad()
        target_texture.data.clamp_(0, 1)

        if iter >= args.base_iters:
            layers = fineTune_layers
        else:
            layers = base_layers

        target_feats = extractor(target_texture)
        loss = get_loss(
            target_feats, refer_feats,
            [target_progression, refer_progression],
            [target_orientation, refer_orientation],
            layers=layers
        )

        loss.backward()
        optimizer.step()

        if iter % args.log_freq == 0:
            target_texture.data.clamp_(0, 1)
            print(f'scale {target_scale} iter {iter + 1} loss {loss.item()}')
            utils.save_image(args.output_folder + f'/output-scale{target_scale}-iter{iter + 1}.jpg', target_texture)

target_texture.data.clamp_(0, 1)
utils.save_image(OUTPUT_FILE, target_texture)

end = time()
print('Time: {:} minutes'.format((end - start) / 60.0))

