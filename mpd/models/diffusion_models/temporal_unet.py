import einops
import numpy as np
import torch
import torch.nn as nn
from abc import ABC
from einops import rearrange
from torch.nn import DataParallel

from mpd.models.layers.layers import GaussianFourierProjection, Downsample1d, Conv1dBlock, Upsample1d, \
    ResidualTemporalBlock, TimeEncoder, MLP, group_norm_n_groups, LinearAttention, PreNorm, Residual, TemporalBlockMLP
from mpd.models.layers.layers_attention import SpatialTransformer


UNET_DIM_MULTS = {
    0: (1, 2, 4),
    1: (1, 2, 4, 8)
}


class TemporalUnet(nn.Module):

    def __init__(
            self,
            n_support_points=None,
            state_dim=None,
            unet_input_dim=32,
            dim_mults=(1, 2, 4, 8),
            time_emb_dim=32,
            self_attention=False,
            conditioning_embed_dim=4,
            conditioning_type=None,
            attention_num_heads=2,
            attention_dim_head=32,
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        input_dim = state_dim

        # Conditioning
        if conditioning_type is None or conditioning_type == 'None':
            conditioning_type = None
        elif conditioning_type == 'concatenate':
            if self.state_dim < conditioning_embed_dim // 4:
                # Embed the state in a latent space HxF if the conditioning embedding is much larger than the state
                state_emb_dim = conditioning_embed_dim // 4
                self.state_encoder = MLP(state_dim, state_emb_dim, hidden_dim=state_emb_dim//2, n_layers=1, act='mish')
            else:
                state_emb_dim = state_dim
                self.state_encoder = nn.Identity()
            input_dim = state_emb_dim + conditioning_embed_dim
        elif conditioning_type == 'attention':
            pass
        elif conditioning_type == 'default':
            pass
        else:
            raise NotImplementedError
        self.conditioning_type = conditioning_type

        dims = [input_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Networks
        self.time_mlp = TimeEncoder(32, time_emb_dim)

        # conditioning dimension (time + context)
        cond_dim = time_emb_dim + (conditioning_embed_dim if conditioning_type == 'default' else 0)

        # Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, cond_dim, n_support_points=n_support_points),
                ResidualTemporalBlock(dim_out, dim_out, cond_dim, n_support_points=n_support_points),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if self_attention else nn.Identity(),
                SpatialTransformer(dim_out, attention_num_heads, attention_dim_head, depth=1,
                                   context_dim=conditioning_embed_dim) if conditioning_type == 'attention' else None,
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                n_support_points = n_support_points // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=n_support_points)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if self_attention else nn.Identity()
        self.mid_attention = SpatialTransformer(mid_dim, attention_num_heads, attention_dim_head, depth=1,
                                                context_dim=conditioning_embed_dim) if conditioning_type == 'attention' else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=n_support_points)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim, n_support_points=n_support_points),
                ResidualTemporalBlock(dim_in, dim_in, cond_dim, n_support_points=n_support_points),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self_attention else nn.Identity(),
                SpatialTransformer(dim_in, attention_num_heads, attention_dim_head, depth=1,
                                   context_dim=conditioning_embed_dim) if conditioning_type == 'attention' else None,
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                n_support_points = n_support_points * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(unet_input_dim, unet_input_dim, kernel_size=5, n_groups=group_norm_n_groups(unet_input_dim)),
            nn.Conv1d(unet_input_dim, state_dim, 1),
        )

    def forward(self, x, time, context):
        """
        x : [ batch x horizon x state_dim ]
        context: [batch x context_dim]
        """
        b, h, d = x.shape

        t_emb = self.time_mlp(time)
        c_emb = t_emb
        if self.conditioning_type == 'concatenate':
            x_emb = self.state_encoder(x)
            context = einops.repeat(context, 'm n -> m h n', h=h)
            x = torch.cat((x_emb, context), dim=-1)
        elif self.conditioning_type == 'attention':
            # reshape to keep the interface
            context = einops.rearrange(context, 'b d -> b 1 d')
        elif self.conditioning_type == 'default':
            c_emb = torch.cat((t_emb, context), dim=-1)

        # swap horizon and channels (state_dim)
        x = einops.rearrange(x, 'b h c -> b c h')  # batch, horizon, channels (state_dim)

        h = []
        for resnet, resnet2, attn_self, attn_conditioning, downsample in self.downs:
            x = resnet(x, c_emb)
            # if self.conditioning_type == 'attention':
            #     x = attention1(x, context=conditioning_emb)
            x = resnet2(x, c_emb)
            x = attn_self(x)
            if self.conditioning_type == 'attention':
                x = attn_conditioning(x, context=context)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c_emb)
        x = self.mid_attn(x)
        if self.conditioning_type == 'attention':
            x = self.mid_attention(x, context=context)
        x = self.mid_block2(x, c_emb)

        for resnet, resnet2, attn_self, attn_conditioning, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c_emb)
            x = resnet2(x, c_emb)
            x = attn_self(x)
            if self.conditioning_type == 'attention':
                x = attn_conditioning(x, context=context)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b c h -> b h c')

        return x


class EnvModel(nn.Module):

    def __init__(
            self,
            in_dim=16,
            out_dim=16,
            **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Identity()

    def forward(self, input_d):
        env = input_d['env']
        env_emb = self.net(env)
        return env_emb


class TaskModel(nn.Module):

    def __init__(
            self,
            in_dim=16,
            out_dim=32,
            **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Identity()

    def forward(self, input_d):
        task = input_d['tasks']
        task_emb = self.net(task)
        return task_emb


class TaskModelNew(nn.Module):

    def __init__(
            self,
            in_dim=16,
            out_dim=32,
            **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Identity()

    def forward(self, task):
        task_emb = self.net(task)
        return task_emb


class ContextModel(nn.Module):

    def __init__(
            self,
            env_model=None,
            task_model=None,
            out_dim=32,
            **kwargs
    ):
        super().__init__()

        self.env_model = env_model
        self.task_model = task_model

        self.in_dim = self.env_model.out_dim + self.task_model.out_dim

        # self.out_dim = out_dim
        # self.net = MLP(self.in_dim, self.out_dim, hidden_dim=out_dim, n_layers=1, act='mish')

        self.out_dim = self.in_dim
        self.net = nn.Identity()

    def forward(self, input_d=None):
        if input_d is None:
            return None
        env_emb = self.env_model(input_d)
        task_emb = self.task_model(input_d)
        context = torch.cat((env_emb, task_emb), dim=-1)
        context_emb = self.net(context)
        return context_emb


class PointUnet(nn.Module):

    def __init__(
            self,
            n_support_points=None,
            state_dim=None,
            dim=32,
            dim_mults=(1, 2, 4),
            time_emb_dim=32,
            conditioning_embed_dim=4,
            conditioning_type=None,
            **kwargs
    ):
        super().__init__()

        self.dim_mults = dim_mults

        self.state_dim = state_dim
        input_dim = state_dim

        # Conditioning
        if conditioning_type is None or conditioning_type == 'None':
            conditioning_type = None
        elif conditioning_type == 'concatenate':
            if self.state_dim < conditioning_embed_dim // 4:
                # Embed the state in a latent space HxF if the conditioning embedding is much larger than the state
                state_emb_dim = conditioning_embed_dim // 4
                self.state_encoder = MLP(state_dim, state_emb_dim, hidden_dim=state_emb_dim//2, n_layers=1, act='mish')
            else:
                state_emb_dim = state_dim
                self.state_encoder = nn.Identity()
            input_dim = state_emb_dim + conditioning_embed_dim
        elif conditioning_type == 'default':
            pass
        else:
            raise NotImplementedError
        self.conditioning_type = conditioning_type

        dims = [input_dim, *map(lambda m: dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Networks
        self.time_mlp = TimeEncoder(32, time_emb_dim)

        # conditioning dimension (time + context)
        cond_dim = time_emb_dim + (conditioning_embed_dim if conditioning_type == 'default' else 0)

        # Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                TemporalBlockMLP(dim_in, dim_out, cond_dim)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = TemporalBlockMLP(mid_dim, mid_dim, cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                TemporalBlockMLP(dim_out * 2, dim_in, cond_dim)
            ]))

        self.final_layer = nn.Sequential(
            MLP(dim, state_dim, hidden_dim=dim, n_layers=0, act='identity')
        )

    def forward(self, x, time, context):
        """
        x : [ batch x horizon x state_dim ]
        context: [batch x context_dim]
        """
        x = einops.rearrange(x, 'b 1 d -> b d')

        t_emb = self.time_mlp(time)
        c_emb = t_emb
        if self.conditioning_type == 'concatenate':
            x_emb = self.state_encoder(x)
            x = torch.cat((x_emb, context), dim=-1)
        elif self.conditioning_type == 'default':
            c_emb = torch.cat((t_emb, context), dim=-1)

        h = []
        for resnet, in self.downs:
            x = resnet(x, c_emb)
            h.append(x)

        x = self.mid_block1(x, c_emb)

        for resnet, in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c_emb)

        x = self.final_layer(x)

        x = einops.rearrange(x, 'b d -> b 1 d')

        return x
