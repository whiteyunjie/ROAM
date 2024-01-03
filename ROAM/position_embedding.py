import math
import torch


def positionalencoding1d(d_model, length, ratio=1):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: (length+1)*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length+1, d_model)
    position = torch.arange(0, length+1).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))*ratio
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width, ratio=1):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(height*width+1, d_model)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    
    height_pe = positionalencoding1d(d_model, height, ratio)
    width_pe = positionalencoding1d(d_model, width, ratio)

    #print(height_pe.shape, width_pe.shape)

    pe[0, :d_model] = height_pe[0]
    pe[0, d_model:] = width_pe[0]

    for i in range(height):
        for j in range(width):
            pe[i*width+j+1, :d_model] = height_pe[i+1]
            pe[i*width+j+1, d_model:] = width_pe[j+1]

    return pe

   
if __name__ == '__main__':
    x20 = positionalencoding2d(512, 8, 8)
    x10 = positionalencoding2d(512, 4, 4, 2)
    x5 = positionalencoding2d(512, 2, 2, 4)
    print(x20, x10, x5)
    cos = torch.nn.CosineSimilarity()
    print(cos(x10[1:2], x20)[1:].reshape((8,8)))


