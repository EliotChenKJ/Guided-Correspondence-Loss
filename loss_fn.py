"""
Implement the slicing operation in PyTorch
"""
import torch
import torch.nn.functional as F
import sys
import math

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
    

# UTILS
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

def compute_orient_distance(x, y, patch_size):
    x = x.view(x.shape[0], 2, patch_size ** 2, x.shape[-2], x.shape[-1])
    y = y.view(y.shape[0], 2, patch_size ** 2, y.shape[-2], y.shape[-1])

    dist = 0
    for i in range(patch_size ** 2):
        dist += torch.min(
            compute_l2_distance(x[:, :, i], y[:, :, i]),
            compute_l2_distance(x[:, :, i], -y[:, :, i])
        )
    dist /= patch_size ** 2

    return dist