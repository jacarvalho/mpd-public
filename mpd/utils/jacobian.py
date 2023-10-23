import torch
from torch import autograd


def get_jacobian(net, x, output_dims, reshape_flag=True, context=None):
    if x.ndimension() == 1:
        n = 1
    else:
        n = x.size()[:-1]
    x_m = x.repeat([1] * len(n) + [output_dims]).view(-1, output_dims)

    x_m.requires_grad_(True)
    y_m = net(x_m)
    mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
    # y.backward(mask)
    J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
    if reshape_flag:
        J = J.reshape(n, output_dims, output_dims)
    return J
