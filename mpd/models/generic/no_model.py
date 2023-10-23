import torch.nn as nn
class NoModel(nn.Module):
    """
    Does nothing. Simply acts as a placeholder to keep the interface.
    """

    def __init__(self, in_dim=16, out_dim=16, input_field='x',
                 output_field='y', **kwargs):
        self.input_field = input_field
        self.output_field = output_field

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, input):
        return input
