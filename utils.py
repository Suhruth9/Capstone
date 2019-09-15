import torch
import torch.nn.functional as F
import numpy as np


def get_size(tensor):
    return(list(tensor.size()))

##def skew_features(features):
##
##    bs, channel, height, width = get_size(features)
##
##    rows = torch.split(features, 1, 2)
##    skewed_rows = []
##
##    skewed_width = width+height-1
##
##    for i, row in enumerate(rows):
##        row = torch.reshape(torch.squeeze(row, dim=2), [-1, width])
##        row = F.pad(row, (i, width-i-1), "constant", 0)
##
##        row = torch.reshape(row, [bs, channel, skewed_width])
##
##        assert get_size(row) == [bs, channel, skewed_width], "undesired skewed row shape"
##        skewed_rows.append(row)
##
##    skewed_features = torch.stack(skewed_rows)
##    assert get_size(skewed_features) == [bs, channel, height, skewed_width], "undesired skewed features shape"
##
##    return skewed_features



def skew_features(features):

    bs, channel, height, width = get_size(features)

    rows = torch.split(features, 1, 2)
    skewed_rows = []

    skewed_width = []

    for i, row in enumerate(rows):
        row = F.pad(row, (0, 0, 0, 0, 0, 0, i, width-i-1), "constant", 0)

        assert get_size(row) == [bs, channel, 1, skewed_width], "undesired row size"
        skewed_rows.append(row)

    skewed_features = torch.cat(skewed_rows, 2, out=skewed_features)
    assert get_size(skewed_features) == [bs, channel, height, skewed_width], "undesired skewed feature size"

    return skewed_features       



def unskew_features(features, width):
    bs, channel, height, width = get_size(features)

    rows = torch.split(features, 1, 2)
    unskewed_rows = []

    unskewed_width = width if width else height

    for i, row in enumerate(rows):
        unskewed_row.append(row[:, :, :, i:width])

    unskewed_features = torch.cat(unskewed_rows, 2, out=unskewed_rows)
    assert get_size(unskewed_features) == [bs, channel, height, width], "undesired unskewed feature size"

    return unskewed_features


def mask( filter_size, input_dim, output_dim, mask_type):

    if isinstance(filter_size, tuple):
        mask = np.ones(output_dim, input_dim, filter_size[0], filter_size[1])
        center_h, center_w = filter_size[0]//2, filter_size[1]//2

    else:
        mask = np.ones(output_dim, input_dim, filter_size, filter_size)
        center_h, center_w = filter_size//2
        

    if max(filter_size)>1:
        mask[:, :, center_h:, center_w+1:] = 0
        mask[:, :, center_h+1:, :] = 0

    if mask_type == "A":
        for i in range(num_channels):
            for j in range(i+1):
                mask[j::num_channels, i::num_channels, center_h, center_w] = 0
                
    elif mask_type == "B":
        for i in range(num_channels):
            for j in range(i):
                mask[j::num_channels, i::num_channels, center_h, center_w] = 0        

    else:
        raise AttributeError("Masktype %s invalid"%mask_type)


    return mask




    

    
        

    


    
    
    







        

    
    

    
    
    
    
    
