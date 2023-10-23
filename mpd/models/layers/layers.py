import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


## Fully connected Neural Network block - Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=16, n_layers=1, act='relu', batch_norm=True):
        super(MLP, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
                       'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus, 'mish': nn.Mish,
                       'identity': nn.Identity
                       }

        act_func = activations[act]
        layers = [nn.Linear(in_dim, hidden_dim), act_func()]
        for i in range(n_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                act_func(),
            ]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self._network = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self._network(x)


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    '''
    Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# https://gist.github.com/kevinzakka/dd9fa5177cda13593524f4d71eb38ad5
class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].
    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.
    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=False):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

        self.temperatur = nn.Parameter(torch.ones(1), requires_grad=True)

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape

        # x = x * h * w
        x = x * (h * w / self.temperatur)
        # print(self.temperatur)
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        xc, yc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        y_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        x_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C*2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


########################################################################################################################
# Modules Temporal Unet
########################################################################################################################
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)


class TimeEncoder(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.encoder = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim_out)
        )

    def forward(self, x):
        return self.encoder(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, padding=None, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, stride=1,
                      padding=padding if padding is not None else kernel_size // 2),
            Rearrange('batch channels n_support_points -> batch channels 1 n_support_points'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 n_support_points -> batch channels n_support_points'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    ################
    # Janner code
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, cond_embed_dim, n_support_points, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, n_groups=group_norm_n_groups(out_channels)),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=group_norm_n_groups(out_channels)),
        ])

        # Without context conditioning, cond_mlp handles only time embeddings
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, c):
        '''
            x : [ batch_size x inp_channels x n_support_points ]
            c : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x n_support_points ]
        '''
        h = self.blocks[0](x) + self.cond_mlp(c)
        h = self.blocks[1](h)
        res = self.residual_conv(x)
        out = h + res

        return out


class TemporalBlockMLP(nn.Module):

    def __init__(self, inp_channels, out_channels, cond_embed_dim):
        super().__init__()

        self.blocks = nn.ModuleList([
            MLP(inp_channels, out_channels, hidden_dim=out_channels, n_layers=0, act='mish')
        ])

        # Without context conditioning, cond_mlp handles only time embeddings
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_embed_dim, out_channels),
            # Rearrange('batch t -> batch t 1'),
        )

        self.last_act = nn.Mish()

    def forward(self, x, c):
        '''
            x : [ batch_size x inp_channels x n_support_points ]
            c : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x n_support_points ]
        '''
        h = self.blocks[0](x) + self.cond_mlp(c)
        out = self.last_act(h)
        return out



def group_norm_n_groups(n_channels, target_n_groups=8):
    if n_channels < target_n_groups:
        return 1
    for n_groups in range(target_n_groups, target_n_groups + 10):
        if n_channels % n_groups == 0:
            return n_groups
    return 1


def compute_padding_conv1d(L, KSZ, S, D, deconv=False):
    '''
    https://gist.github.com/AhmadMoussa/d32c41c11366440bc5eaf4efb48a2e73
    :param L:       Input length (or width)
    :param KSZ:     Kernel size (or width)
    :param S:       Stride
    :param D:       Dilation Factor
    :param deconv:  True if ConvTranspose1d
    :return:        Returns padding such that output width is exactly half of input width
    '''
    print(f"INPUT SIZE {L}")
    if not deconv:
        return math.ceil((S * (L / 2) - L + D * (KSZ - 1) - 1) / 2)
    else:
        print(L, S, D, KSZ)
        pad = math.ceil(((L - 1) * S + D * (KSZ - 1) + 1 - L * 2) / 2)
        print("PAD", pad)
        output_size = (L - 1) * S - 2 * pad + D * (KSZ - 1) + 1
        print("OUTPUT SIZE", output_size)
        return pad


def compute_output_length_maxpool1d(L, KSZ, S, D, P):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    :param L:       Input length (or width)
    :param KSZ:     Kernel size (or width)
    :param S:       Stride
    :param D:       Dilation Factor
    :param P:       Padding
    '''
    return math.floor((L + 2 * P - D * (KSZ - 1) - 1) / S + 1)


if __name__ == "__main__":
    b, c, h, w = 1, 64, 12, 12
    x = torch.full((b, c, h, w), 0.00)

    i_max = 4
    true_max = torch.randint(0, 10, size=(b, c, 2))
    for i in range(b):
        for j in range(c):
            x[i, j, true_max[i, j, 0], true_max[i, j, 1]] = 1000
        # x[i, j, i_max, true_max] = 1
    # x[0,0,0,0] = 1000
    soft_max = SpatialSoftArgmax(normalize=True)(x)
    soft_max2 = SpatialSoftArgmax(normalize=False)(x)
    diff = soft_max - (soft_max2 / 12) * 2 - 1
    resh = soft_max.reshape(b, c, 2)
    assert torch.allclose(true_max.float(), resh)

    exit()
    test_scales = [1, 5, 10, 30, 50, 70, 100]
    for scale in test_scales[::-1]:
        i_max = 4
        true_max = torch.randint(0, 10, size=(b, c, 2))
        for i in range(b):
            for j in range(c):
                x[i, j, true_max[i, j, 0], true_max[i, j, 1]] = scale
            # x[i, j, i_max, true_max] = 1
        # x[0,0,0,0] = 1000
        soft_max = SpatialSoftArgmax(normalize=False)(x)
        resh = soft_max.reshape(b, c, 2)
        assert torch.allclose(true_max.float(), resh), scale
