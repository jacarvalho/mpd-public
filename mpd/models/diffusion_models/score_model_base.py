import torch.nn as nn
from abc import ABC
from mpd.models.helpers.marginal_prob import reshape_std


class ScoreModelBase(nn.Module, ABC):

    def __init__(self,
                 marginal_prob_get_std=None,
                 input_dim=None,
                 context_dim=0,
                 input_field='x',
                 output_field='grad_x_log_p_x',
                 context_field='context',
                 **kwargs):

        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim = input_dim  # by default, score models output's dimension is the same as the input
        self.marginal_prob_get_std = marginal_prob_get_std
        self.input_field = input_field
        self.output_field = output_field
        self.context_field = context_field

    def forward(self, input_dict):
        out = self.compute_unnormalized_score(input_dict)
        if type(out) is dict:
            out_score = out[self.output_field]
        else:
            out_score = out
            out = {}
        # Normalize output
        t = input_dict['t']
        std = reshape_std(self.marginal_prob_get_std(t), out_score)
        out_score = out_score / std
        return {**out, self.output_field: out_score}

    def compute_unnormalized_score(self, data):
        raise NotImplementedError

