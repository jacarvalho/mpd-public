import numpy as np
import torch
import torch.nn as nn

from mpd.models.layers.layers import GaussianFourierProjection, Downsample1d, Conv1dBlock, Upsample1d, \
    ResidualTemporalBlock, TimeEncoder, MLP
from mpd.models.layers.layers_attention import SpatialTransformer
from mpd.models.helpers.marginal_prob import reshape_std
from .score_model_base import ScoreModelBase
class MLPScoreModel(ScoreModelBase):
    '''
    Simple model using dictionaries
    '''

    def __init__(self, marginal_prob_get_std=None,
                 input_dim=None, hidden_dim=None, context_embed_dim=0,
                 time_embed_dim=10,
                 act='relu',
                 input_field='x',
                 output_field='dx',
                 context_field='c',
                 n_layers=4, **kwargs):
        '''

        Args:
            in_dim:
            out_dim:
            hidden_dim:
            marginal_prob_std:
            time_embed_dim:
            act:
            input_field:
            context_field: Context to be used (e.g. state) if None then it's excluded
        '''
        super(MLPScoreModel, self).__init__(
            marginal_prob_get_std,
            input_dim,
            context_embed_dim,
            input_field,
            output_field,
            context_field
        )
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
                       'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

        self.act_func = activations[act]

        # Gaussian random feature embedding layer for time
        self.time_embedding = nn.Sequential(GaussianFourierProjection(embed_dim=time_embed_dim),
                                            nn.Linear(time_embed_dim, time_embed_dim), self.act_func())

        self.n_layers = n_layers

        layers = [nn.Linear(time_embed_dim + context_embed_dim + np.prod(input_dim), hidden_dim), self.act_func()]
        for n in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.act_func())
        layers.append(nn.Linear(hidden_dim, np.prod(input_dim)))
        self._net = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def compute_unnormalized_score(self, input_dict):
        x = self.flatten(input_dict[self.input_field])
        t = input_dict['t']
        ## Embed time
        # Obtain the Gaussian random feature embedding for t
        time_embed = self.time_embedding(t)
        ## Embed Input
        if self.context_field in input_dict:
            context = input_dict[self.context_field]
            x_t = torch.cat((x, context, time_embed), dim=1)
        else:
            x_t = torch.cat((x, time_embed), dim=1)
        out = self._net(x_t)

        return out


class ContextScoreModel(MLPScoreModel):
    def __init__(self,
                 env_model=None,
                 task_model=None,
                 initial_config_field=None,
                 initial_config_dim=0,
                 **kwargs):
        context_embed_dim = env_model.out_dim + task_model.out_dim + initial_config_dim
        super(ContextScoreModel, self).__init__(**kwargs, context_embed_dim=context_embed_dim)
        self.env_model = env_model
        self.task_model = task_model

        self.initial_config_field = initial_config_field
        self.initial_config_dim = initial_config_dim

    def forward_old(self, input):
        env = self.env_model(input)[self.env_model.output_field]
        task = self.task_model(input)[self.task_model.output_field]
        # env = input[self.env_model.output_field]
        # tasks = input[self.task_model.output_field]
        start = input[self.initial_config_field]
        context_embed = torch.cat((env, start, task), dim=1)

        score_model_input = {**input, self.context_field: context_embed}
        return self._forward(score_model_input)

    def compute_unnormalized_score(self, input_dict):
        x = self.flatten(input_dict[self.input_field])
        t = input_dict['t']

        # Embed Context
        env = self.env_model(input_dict)[self.env_model.output_field]
        task = self.task_model(input_dict)[self.task_model.output_field]
        start = input_dict[self.initial_config_field]

        ## Embed time
        # Obtain the Gaussian random feature embedding for t
        time_embed = self.time_embedding(t)

        xt = torch.cat((x, env, task, start, time_embed), dim=1)
        out = self._net(xt)

        return {self.output_field: out}


class SDFScoreModel(MLPScoreModel):
    def __init__(self,
                 env_model=None,
                 task_model=None,
                 sdf_model=None,
                 initial_config_field=None,
                 initial_config_dim=0,
                 use_sdf_value=False,
                 n_support_points=20,
                 **kwargs):
        context_embed_dim = env_model.out_dim + task_model.out_dim + \
                            (n_support_points * sdf_model.out_dim if use_sdf_value else 0)
        super(SDFScoreModel, self).__init__(**kwargs, context_embed_dim=context_embed_dim)
        self.env_model = env_model
        self.task_model = task_model
        self.sdf_model = sdf_model

        self.use_sdf = use_sdf_value
        self.n_support_points = n_support_points

        self.initial_config_field = initial_config_field
        self.initial_config_dim = initial_config_dim

    def compute_unnormalized_score(self, input_dict):
        orig_shape = input_dict[self.input_field].shape
        x = self.flatten(input_dict[self.input_field])
        t = input_dict['t']

        if self.sdf_model.sdf_location_field not in input_dict:
            batch_size, flat = x.shape
            locations = x.reshape(batch_size, flat // 2, 2)[:, 0, :].squeeze(1)
            input_dict[self.sdf_model.sdf_location_field] = locations

        # Embed Context
        env = self.env_model(input_dict)[self.env_model.output_field]
        task = self.task_model(input_dict)[self.task_model.output_field]
        # start = input_dict[self.initial_config_field]

        # Run SDF
        sdf = self.sdf_model({**input_dict, self.sdf_model.input_field: env})

        ## Embed time
        # Obtain the Gaussian random feature embedding for t
        time_embed = self.time_embedding(t)

        xt = torch.cat((x, env, task, time_embed), dim=1)
        out = self._net(xt)
        out = out.reshape(orig_shape)

        return {**sdf, self.output_field: out}


class SharedFeatureSDFScoreModel(MLPScoreModel):
    def __init__(self,
                 env_model=None,
                 task_model=None,
                 sdf_model=None,
                 initial_config_field=None,
                 initial_config_dim=0,
                 use_sdf_value=False,
                 n_support_points=20,
                 **kwargs):
        context_embed_dim = sdf_model.hidden_dim + task_model.out_dim + initial_config_dim + \
                            (n_support_points * sdf_model.out_dim if use_sdf_value else 0)
        super(SharedFeatureSDFScoreModel, self).__init__(**kwargs, context_embed_dim=context_embed_dim)
        self.env_model = env_model
        self.task_model = task_model
        self.sdf_model = sdf_model

        self.use_sdf = use_sdf_value
        self.n_support_points = n_support_points

        self.initial_config_field = initial_config_field
        self.initial_config_dim = initial_config_dim

    def compute_unnormalized_score(self, input_dict):
        x = self.flatten(input_dict[self.input_field])
        t = input_dict['t']

        # Embed Context
        env = self.env_model(input_dict)[self.env_model.output_field]
        task = self.task_model(input_dict)[self.task_model.output_field]
        start = input_dict[self.initial_config_field]

        # Run SDF
        sdf = self.sdf_model({**input_dict, self.sdf_model.input_field: env,
                              self.sdf_model.sdf_location_field: x})
        sdf_features = sdf[self.sdf_model.feature_field]

        ## Embed time
        # Obtain the Gaussian random feature embedding for t
        time_embed = self.time_embedding(t)

        xt = torch.cat((x, sdf_features, task, start, time_embed), dim=1)
        out = self._net(xt)

        return {**sdf, self.output_field: out}
