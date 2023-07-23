"""
Implement the slicing operation in PyTorch
"""
import torch
import torch.nn.functional as F
import sys
import math


class Slicing_torch(torch.nn.Module):
    def __init__(self, device, layers, guidance=None):
        super().__init__()
        # Number of directions
        self.device = device
        self.update_slices(layers, guidance)

        # self.patch_size = patch_size

    def update_slices(self, layers, guidance=None):
        # add random direction
        directions = []
        concat_layers = []

        layers_len = len(layers)
        # self.used_layers = [i for i in range(layers_len-1, layers_len-10, -1)]

        for index, l in enumerate(layers):
            # converted to [B, W, H, D]
            if l.ndim == 4:
                l = l.permute(0, 2, 3, 1)
            if l.ndim == 5:
                l = l.permute(0, 2, 3, 4, 1)

            # random direction
            dim_slices = l.shape[-1]
            num_slices = l.shape[-1]
            cur_dir = torch.randn(size=(num_slices, dim_slices)).to(self.device)
            norm = torch.sqrt(torch.sum(torch.square(cur_dir), 1, keepdim=True))
            cur_dir = cur_dir / norm

            # extra dimension for direction
            if type(guidance) is not type(None):
                extra_dim_slices = guidance.shape[1]
                one_dir = torch.ones(size=(num_slices, extra_dim_slices)).to(self.device)
                cur_dir = torch.cat([cur_dir, one_dir], dim=-1)
                cur_guidance = F.interpolate(guidance, l.shape[1:3])
                concat_layers.append(torch.cat([layers[index], cur_guidance], 1))
            else:
                concat_layers.append(layers[index])
            directions.append(cur_dir)
        self.directions = directions

        # compute sorted target features
        self.target = self.compute_target(concat_layers)

    def compute_proj(self, input, layer_idx):
        if input.ndim == 4:
            input = input.permute(0, 2, 3, 1)
        if input.ndim == 5:
            input = input.permute(0, 2, 3, 4, 1)

        batch = input.size(0)
        dim = input.size(-1)
        tensor = input.view(batch, -1, dim)
        tensor_permute = tensor.permute(0, 2, 1)

        # Project each pixel feature onto directions (batch dot product)
        sliced = torch.matmul(self.directions[layer_idx], tensor_permute)

        # # Sort projections for each direction
        sliced, _ = torch.sort(sliced)
        sliced = sliced.view(batch, -1)
        return sliced

    def compute_target(self, layers):
        target = []
        for idx, l in enumerate(layers):
            sliced_l = self.compute_proj(l, idx)
            target.append(sliced_l.detach())
        return target

    def forward(self, layers, guidance=None):
        loss = 0.0
        # use extra guidance or not
        if type(guidance) is not type(None):
            input = []
            for index, layer in enumerate(layers):
                cur_guidance = F.interpolate(guidance, layer.shape[-2:])
                input.append(torch.cat([layer, cur_guidance], 1))
        else:
            input = layers

        # compute loss
        for idx, l in enumerate(input):
            cur_l = self.compute_proj(l, idx)
            tar_l = self.target[idx]
            if cur_l.shape != tar_l.shape: # upsample if neccessary
                tar_l = F.interpolate(tar_l[:, None, :], size=cur_l.shape[-1])[:, 0, :]
            loss += F.mse_loss(cur_l, tar_l)
        return loss


# contextual loss related
def isNone(x):
    return type(x) is type(None)


def feature_normalize(feature_in):
    feature_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_norm)
    return feature_in_norm, feature_norm


def batch_patch_extraction(image_tensor, kernel_size, stride):
    """ [n, c, h, w] -> [n, np(num_patch), c, k, k] """
    n, c, h, w = image_tensor.shape
    h_out = math.floor((h - (kernel_size-1) - 1) / stride + 1)
    w_out = math.floor((w - (kernel_size-1) - 1) / stride + 1)
    unfold_tensor = F.unfold(image_tensor, kernel_size=kernel_size, stride=stride)
    unfold_tensor = unfold_tensor.contiguous().view(
        n, c * kernel_size * kernel_size, h_out, w_out)
    return unfold_tensor


def compute_cosine_distance(x, y):
    N, C, _, _ = x.size()

    # to normalized feature vectors
    # x_mean = x.view(N, C, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    y_mean = y.view(N, C, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    x, x_norm = feature_normalize(x - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    y, y_norm = feature_normalize(y - y_mean)  # batch_size * feature_depth * feature_size * feature_size
    x = x.view(N, C, -1)
    y = y.view(N, C, -1)

    # cosine distance = 1 - similarity
    x_permute = x.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth

    # convert similarity to distance
    sim = torch.matmul(x_permute, y)
    dist = (1 - sim) / 2 # batch_size * feature_size^2 * feature_size^2

    return dist.clamp(min=0.)


def compute_l2_distance(x, y):
    N, C, Hx, Wx = x.size()
    _, _, Hy, Wy = y.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s.unsqueeze(2).expand_as(A) - 2 * A + x_s.unsqueeze(1).expand_as(A)
    dist = dist.transpose(1, 2).reshape(N, Hx*Wx, Hy*Wy)
    dist = dist.clamp(min=0.) / C

    return dist


class GuidedCorrespondenceLoss_forward(torch.nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self,
                 sample_size=100, h=0.5, patch_size=7,
                 progression_weight=0,
                 orientation_weight=0,
                 occurrence_weight=0):
        super(GuidedCorrespondenceLoss_forward, self).__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = max(patch_size // 2, 2)
        self.h = h

        self.progression_weight = progression_weight
        self.orientation_weight = orientation_weight
        self.occurrence_weight = occurrence_weight

    def feature_extraction(self, feature, sample_field=None):
        # Patch extraction - use patch as single feature
        if self.patch_size > 1:
            feature = batch_patch_extraction(feature, self.patch_size, self.stride)

        # Random sampling - random patches
        num_batch, num_channel = feature.shape[:2]
        if num_batch * feature.shape[-2] * feature.shape[-1] > self.sample_size ** 2:
            if isNone(sample_field):
                sample_field = torch.rand(
                    num_batch, self.sample_size, self.sample_size, 2, device=feature.device) * 2 - 1
            feature = F.grid_sample(feature, sample_field, mode='nearest')

        # Concatenate tensor
        sampled_feature = feature

        return sampled_feature, sample_field

    def calculate_distance(self, target_features, refer_features, progressions=None, orientations=None):
        origin_target_size = target_features.shape[-2:]
        origin_refer_size = refer_features.shape[-2:]

        # feature
        target_features, target_field = self.feature_extraction(target_features)
        refer_features, refer_field = self.feature_extraction(refer_features)
        d_total = compute_cosine_distance(target_features, refer_features)

        # progression
        use_progression = self.progression_weight > 0 and not isNone(progressions)
        if use_progression:
            with torch.no_grad():
                target_prog, refer_prog = progressions  # resize progression to corresponding size
                target_prog = F.interpolate(target_prog, origin_target_size)
                refer_prog = F.interpolate(refer_prog, origin_refer_size)

                target_prog = self.feature_extraction(target_prog, target_field)[0]
                refer_prog = self.feature_extraction(refer_prog, refer_field)[0]

                d_prog = compute_l2_distance(target_prog, refer_prog)
            d_total += d_prog * self.progression_weight

        # orientation
        use_orientation = self.orientation_weight > 0 and not isNone(orientations)
        if use_orientation:
            with torch.no_grad():
                target_orient, refer_orient = orientations
                target_orient = F.interpolate(target_orient, origin_target_size)
                refer_orient = F.interpolate(refer_orient, origin_refer_size)

                target_orient = self.feature_extraction(target_orient, target_field)[0]
                refer_orient = self.feature_extraction(refer_orient, refer_field)[0]
                target_orient = target_orient.view(target_orient.shape[0], 2, self.patch_size ** 2,
                                                   target_orient.shape[-2], target_orient.shape[-1])
                refer_orient = refer_orient.view(refer_orient.shape[0], 2, self.patch_size ** 2,
                                                 refer_orient.shape[-2], refer_orient.shape[-1])

                d_orient = 0
                for i in range(self.patch_size ** 2):
                    d_orient += torch.min(
                        compute_l2_distance(target_orient[:, :, i], refer_orient[:, :, i]),
                        compute_l2_distance(target_orient[:, :, i], -refer_orient[:, :, i])
                    )
                d_orient /= self.patch_size ** 2
            d_total += d_orient * self.orientation_weight

        min_idx_for_target = torch.min(d_total, dim=-1, keepdim=True)[1]

        # occurrence penalty
        use_occurrence = self.occurrence_weight > 0
        if use_occurrence:
            with torch.no_grad():
                omega = d_total.shape[1] / d_total.shape[2]
                occur = torch.zeros_like(d_total[:, 0, :])
                indexs, counts = min_idx_for_target[0, :, 0].unique(return_counts=True)
                occur[:, indexs] = counts / omega
                occur = occur.view(1, 1, -1)
                d_total += occur * self.occurrence_weight

        # propagation penalty
        # use_propagation = True
        # if use_propagation:
        #     with torch.no_grad():
        #         curr_target_size = target_features.shape[-2:]
        #         curr_refer_size = refer_features.shape[-2:]
        #         closest_refer = min_idx_for_target.view(min_idx_for_target.shape[0], curr_target_size[0], curr_target_size[1])
        #
        #         closest_refer_y = closest_refer // curr_refer_size[1]
        #         closest_refer_y = torch.cat([closest_refer_y[:, :1, :] - 1, closest_refer_y, closest_refer_y[:, -1:, :] + 1], 1)
        #         penalty_y = (torch.abs(closest_refer_y[:, 1:-1] - closest_refer_y[:, :-2] - 1) + \
        #                      torch.abs(closest_refer_y[:, 2:] - closest_refer_y[:, 1:-1] - 1)) / 2
        #
        #         closest_refer_x = closest_refer % curr_refer_size[1]
        #         closest_refer_x = torch.cat([closest_refer_x[:, :, :1] - 1, closest_refer_x, closest_refer_x[:, :, -1:] + 1], 2)
        #         penalty_x = (torch.abs(closest_refer_x[:, :, 1:-1] - closest_refer_x[:, :, :-2] - 1) + \
        #                      torch.abs(closest_refer_x[:, :, 2:] - closest_refer_x[:, :, 1:-1] - 1)) / 2
        #
        #         prog_penalty = (penalty_y + penalty_x) / 2
        #         prog_penalty = prog_penalty.view(prog_penalty.shape[0], -1, 1) * 0.05
        #         d_total.scatter_add_(-1, min_idx_for_target, prog_penalty)

        # use_propagation = True
        # if use_propagation:
        #     with torch.no_grad():
        #         curr_target_size = target_features.shape[-2:]
        #         curr_refer_size = refer_features.shape[-2:]
        #         closest_refer = min_idx_for_target.view(min_idx_for_target.shape[0], curr_target_size[0], curr_target_size[1])
        #
        #         closest_refer_y = closest_refer // curr_refer_size[1]
        #         closest_refer_y = torch.cat([closest_refer_y[:, :1, :] - 1, closest_refer_y], 1)
        #         penalty_y = torch.abs(closest_refer_y[:, 1:] - closest_refer_y[:, :-1] - 1)
        #
        #         closest_refer_x = closest_refer % curr_refer_size[1]
        #         closest_refer_x = torch.cat([closest_refer_x[:, :, :1] - 1, closest_refer_x], 2)
        #         penalty_x = torch.abs(closest_refer_x[:, :, 1:] - closest_refer_x[:, :, :-1] - 1)
        #
        #         prog_penalty = (penalty_y + penalty_x) / 2
        #         prog_penalty = prog_penalty.view(prog_penalty.shape[0], -1, 1) * 0.05
        #         d_total.scatter_add_(-1, min_idx_for_target, prog_penalty)

        return d_total

    def calculate_loss(self, d):
        # --------------------------------------------------
        # minimize closest distance
        # --------------------------------------------------
        # loss = d.min(dim=-1)[0].mean()

        # --------------------------------------------------
        # guided correspondence loss
        # --------------------------------------------------
        # calculate loss
        # for each target feature, find closest refer feature
        d_min = torch.min(d, dim=-1, keepdim=True)[0]
        # convert to relative distance
        d_norm = d / (d_min + sys.float_info.epsilon)
        w = torch.exp((1 - d_norm) / self.h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        # texture loss per sample
        CX = torch.max(A_ij, dim=-1)[0]
        loss = -torch.log(CX).mean()

        # --------------------------------------------------
        # contextual loss
        # --------------------------------------------------
        # # calculate contextual similarity and contextual loss
        # d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + sys.float_info.epsilon)  # batch_size * feature_size^2 * feature_size^2
        # # pairwise affinity
        # w = torch.exp((1 - d_norm) / self.h)
        # A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        # # contextual loss per sample
        # CX = torch.max(A_ij, dim=1)[0].mean(dim=-1)
        # loss = -torch.log(CX).mean()

        return loss

    def forward(self, target_features, refer_features, progressions=None, orientations=None):
        d_total = self.calculate_distance(
            target_features, refer_features,
            progressions, orientations)
        loss = self.calculate_loss(d_total)

        return loss


class AugmentedGuidedCorrespondenceLoss_forward(torch.nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self,
                 sample_size=100, h=0.5, patch_size=7,
                 progression_weight=0,
                 orientation_weight=0,
                 occurrence_weight=0):
        super(AugmentedGuidedCorrespondenceLoss_forward, self).__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = patch_size // 2 + 1
        self.h = h

        self.progression_weight = progression_weight
        self.orientation_weight = orientation_weight
        self.occurrence_weight = occurrence_weight

    def feature_extraction(self, features):
        sampled_features = []
        sampled_sizes = []

        for idx, feature in enumerate(features):
            # Patch extraction - use patch as single feature
            if self.patch_size > 1:
                feature = batch_patch_extraction(feature, self.patch_size, self.stride)

            sampled_features.append(feature)
            sampled_sizes.append(feature.shape[-2:])

        sampled_feature = torch.cat([feat.view(feat.shape[0], feat.shape[1], 1, -1) for feat in sampled_features], -1)

        return sampled_feature, sampled_sizes

    def calculate_distance(self, target_features, refer_features, progressions=None, orientations=None, refer_coordinates=None):
        origin_target_shapes = [target_feature.shape[-2:] for target_feature in target_features]
        origin_refer_shapes = [refer_feature.shape[-2:] for refer_feature in refer_features]

        # feature
        target_features, target_sizes = self.feature_extraction(target_features)
        refer_features, refer_sizes = self.feature_extraction(refer_features)
        d_total = compute_cosine_distance(target_features, refer_features)

        # progression
        use_progression = self.progression_weight > 0 and not isNone(progressions)
        if use_progression:
            with torch.no_grad():
                target_progs, refer_progs = progressions  # resize progression to corresponding size
                target_progs = [F.interpolate(target_prog, origin_target_shapes[idx]) for idx, target_prog in
                                enumerate(target_progs)]
                refer_progs = [F.interpolate(refer_prog, origin_refer_shapes[idx]) for idx, refer_prog in
                               enumerate(refer_progs)]

                target_prog = self.feature_extraction(target_progs)[0]
                refer_prog = self.feature_extraction(refer_progs)[0]

                d_prog = compute_l2_distance(target_prog, refer_prog)
            d_total += d_prog * self.progression_weight

        # orientation
        use_orientation = self.orientation_weight > 0 and not isNone(orientations)
        if use_orientation:
            with torch.no_grad():
                target_orients, refer_orients = orientations
                target_orients = [F.interpolate(target_orient, origin_target_shapes[idx]) for idx, target_orient in
                                  enumerate(target_orients)]
                refer_orients = [F.interpolate(refer_orient, origin_refer_shapes[idx]) for idx, refer_orient in
                                 enumerate(refer_orients)]

                target_orient = self.feature_extraction(target_orients)[0]
                refer_orient = self.feature_extraction(refer_orients)[0]
                target_orient = target_orient.view(target_orient.shape[0], 2, self.patch_size ** 2,
                                                   target_orient.shape[-2], target_orient.shape[-1])
                refer_orient = refer_orient.view(refer_orient.shape[0], 2, self.patch_size ** 2,
                                                 refer_orient.shape[-2], refer_orient.shape[-1])

                d_orient = 0
                for i in range(self.patch_size ** 2):
                    d_orient += torch.min(
                        compute_l2_distance(target_orient[:, :, i], refer_orient[:, :, i]),
                        compute_l2_distance(target_orient[:, :, i], -refer_orient[:, :, i])
                    )
                d_orient /= self.patch_size ** 2
            d_total += d_orient * self.orientation_weight

        # occurrence penalty
        use_occurrence = self.occurrence_weight > 0
        if use_occurrence:
            with torch.no_grad():
                refer_height, refer_width = refer_sizes[0]
                refer_size = refer_height * refer_width

                refer_coords = []
                for idx, refer_coord in enumerate(refer_coordinates):
                    refer_coord = F.interpolate(refer_coord, refer_sizes[idx],
                        mode='bilinear', align_corners=True)
                    refer_coord = refer_coord[:, 0] * (refer_height - 1) * refer_width + \
                                  refer_coord[:, 1] * (refer_width - 1)
                    refer_coord = refer_coord.round().long()
                    refer_coord = refer_coord.permute(1, 0, 2).flatten()
                    refer_coords.append(refer_coord)
                refer_coord = torch.cat(refer_coords, dim=0)

                omega = d_total.shape[1] / refer_size
                occur = torch.zeros(refer_size).to(refer_coord.device)
                d_min_idx = torch.min(d_total, dim=-1, keepdim=True)[1]
                min_index, counts = refer_coord[d_min_idx[0, :, 0]].unique(return_counts=True)
                occur[min_index] = counts / omega
                occur = occur[refer_coord].view(1, 1, -1)

                d_total += occur * self.occurrence_weight

        # use_propagation = True
        # if use_propagation:
        #     with torch.no_grad():
        #         refer_height, refer_width = refer_sizes[0]
        #
        #         refer_coords = []
        #         for idx, refer_coord in enumerate(refer_coordinates):
        #             refer_coord = F.interpolate(refer_coord, refer_sizes[idx],
        #                                         mode='bilinear', align_corners=True)
        #             refer_coord[:, 0] *= (refer_height - 1)
        #             refer_coord[:, 1] *= (refer_width - 1)
        #             refer_coord = refer_coord.round().long()
        #             refer_coord = refer_coord.view(2, -1).permute(1, 0)
        #             refer_coords.append(refer_coord)
        #         refer_coord = torch.cat(refer_coords, dim=0)
        #
        #         argmin_idx_for_target = torch.min(d_total, dim=-1, keepdim=True)[1].flatten()
        #         closest_refer = refer_coord[argmin_idx_for_target].view(1, *target_sizes[0], 2)
        #
        #         closest_refer_y = closest_refer[:, :, :, 0]
        #         closest_refer_y = torch.cat([closest_refer_y[:, :1, :] - 1, closest_refer_y, closest_refer_y[:, -1:, :] + 1], 1)
        #         y_penalty = torch.abs(closest_refer_y[:, 1:-1] - closest_refer_y[:, :-2] - 1) + \
        #                     torch.abs(closest_refer_y[:, 2:] - closest_refer_y[:, 1:-1] - 1)
        #
        #         closest_refer_x = closest_refer[:, :, :, 1]
        #         closest_refer_x = torch.cat([closest_refer_x[:, :, :1] - 1, closest_refer_x, closest_refer_x[:, :, -1:] + 1], 2)
        #         x_penalty = torch.abs(closest_refer_x[:, :, 1:-1] - closest_refer_x[:, :, :-2] - 1) + \
        #                     torch.abs(closest_refer_x[:, :, 2:] - closest_refer_x[:, :, 1:-1] - 1)
        #
        #         prog_penalty = (y_penalty + x_penalty) / 4
        #         prog_penalty = prog_penalty.view(prog_penalty.shape[0], -1, 1) * 0.1
        #         d_total.scatter_add_(-1, argmin_idx_for_target.view(1, -1, 1), prog_penalty)

        return d_total

    def calculate_loss(self, d):
        # --------------------------------------------------
        # guided correspondence loss
        # --------------------------------------------------
        # calculate loss
        # for each target feature, find closest refer feature
        d_min = torch.min(d, dim=-1, keepdim=True)[0]
        # convert to relative distance
        d_norm = d / (d_min + sys.float_info.epsilon)
        w = torch.exp((1 - d_norm) / self.h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        # texture loss per sample
        CX = torch.max(A_ij, dim=-1)[0]
        loss = -torch.log(CX).mean()

        return loss

    def forward(self, target_features, refer_features, progressions=None, orientations=None, refer_coordinate=None):
        d_total = self.calculate_distance(
            target_features, refer_features,
            progressions, orientations,
            refer_coordinate)
        loss = self.calculate_loss(d_total)

        return loss


class ContextualLoss_forward(torch.nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self,
                 sample_size=100, h=0.5, patch_size=7,
                 progression_weight=0,
                 orientation_weight=0,
                 occurrence_weight=0):
        super(ContextualLoss_forward, self).__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.stride = max(patch_size // 2, 2)
        self.h = h

        self.progression_weight = progression_weight
        self.orientation_weight = orientation_weight
        self.occurrence_weight = occurrence_weight

    def feature_extraction(self, feature, sample_field=None):
        # Patch extraction - use patch as single feature
        if self.patch_size > 1:
            feature = batch_patch_extraction(feature, self.patch_size, self.stride)

        # Random sampling - random patches
        num_batch, num_channel = feature.shape[:2]
        if num_batch * feature.shape[-2] * feature.shape[-1] > self.sample_size ** 2:
            if isNone(sample_field):
                sample_field = torch.rand(
                    num_batch, self.sample_size, self.sample_size, 2, device=feature.device) * 2 - 1
            feature = F.grid_sample(feature, sample_field, mode='nearest')

        # Concatenate tensor
        sampled_feature = feature

        return sampled_feature, sample_field

    def calculate_distance(self, target_features, refer_features, progressions=None, orientations=None):
        origin_target_size = target_features.shape[-2:]
        origin_refer_size = refer_features.shape[-2:]

        # feature
        target_features, target_field = self.feature_extraction(target_features)
        refer_features, refer_field = self.feature_extraction(refer_features)
        d_total = compute_cosine_distance(target_features, refer_features)

        # progression
        use_progression = self.progression_weight > 0 and not isNone(progressions)
        if use_progression:
            with torch.no_grad():
                target_prog, refer_prog = progressions  # resize progression to corresponding size
                target_prog = F.interpolate(target_prog, origin_target_size)
                refer_prog = F.interpolate(refer_prog, origin_refer_size)

                target_prog = self.feature_extraction(target_prog, target_field)[0]
                refer_prog = self.feature_extraction(refer_prog, refer_field)[0]

                d_prog = compute_l2_distance(target_prog, refer_prog)
            d_total += d_prog * self.progression_weight

        # orientation
        use_orientation = self.orientation_weight > 0 and not isNone(orientations)
        if use_orientation:
            with torch.no_grad():
                target_orient, refer_orient = orientations
                target_orient = F.interpolate(target_orient, origin_target_size)
                refer_orient = F.interpolate(refer_orient, origin_refer_size)

                target_orient = self.feature_extraction(target_orient, target_field)[0]
                refer_orient = self.feature_extraction(refer_orient, refer_field)[0]
                target_orient = target_orient.view(target_orient.shape[0], 2, self.patch_size ** 2,
                                                   target_orient.shape[-2], target_orient.shape[-1])
                refer_orient = refer_orient.view(refer_orient.shape[0], 2, self.patch_size ** 2,
                                                 refer_orient.shape[-2], refer_orient.shape[-1])

                d_orient = 0
                for i in range(self.patch_size ** 2):
                    d_orient += torch.min(
                        compute_l2_distance(target_orient[:, :, i], refer_orient[:, :, i]),
                        compute_l2_distance(target_orient[:, :, i], -refer_orient[:, :, i])
                    )
                d_orient /= self.patch_size ** 2
            d_total += d_orient * self.orientation_weight

        min_idx_for_target = torch.min(d_total, dim=-1, keepdim=True)[1]

        # occurrence penalty
        use_occurrence = self.occurrence_weight > 0
        if use_occurrence:
            with torch.no_grad():
                omega = d_total.shape[1] / d_total.shape[2]
                occur = torch.zeros_like(d_total[:, 0, :])
                indexs, counts = min_idx_for_target[0, :, 0].unique(return_counts=True)
                occur[:, indexs] = counts / omega
                occur = occur.view(1, 1, -1)
                d_total += occur * self.occurrence_weight

        return d_total

    def calculate_loss(self, d):
        # --------------------------------------------------
        # contextual loss
        # --------------------------------------------------
        # calculate contextual similarity and contextual loss
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + sys.float_info.epsilon)  # batch_size * feature_size^2 * feature_size^2
        # pairwise affinity
        w = torch.exp((1 - d_norm) / self.h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)
        # contextual loss per sample
        CX = torch.max(A_ij, dim=1)[0].mean(dim=-1)
        loss = -torch.log(CX).mean()

        return loss

    def forward(self, target_features, refer_features, progressions=None, orientations=None):
        d_total = self.calculate_distance(
            target_features, refer_features,
            progressions, orientations)
        loss = self.calculate_loss(d_total)

        return loss
