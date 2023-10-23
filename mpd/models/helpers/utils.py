def calc_output(in_dim, filter_size, padding=0, stride=1):
    x = in_dim - filter_size + (2 * padding) + (filter_size % 2)
    return (x / stride) + (1 - (filter_size % 2))


def calc_output_conv2d_transpose(in_dim, filter_size, padding=0, stride=1, dilation=1):
    return (in_dim - 1) * stride - 2 * padding + dilation * (filter_size - 1) + 1
