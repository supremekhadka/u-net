def crop_feature_map(feature_map, x):
    '''
    Center crops feature map to the shape of x in (h, w) dimensions.
    '''
    diff_h = (feature_map.shape[-2] - x.shape[-2])/2 
    diff_w = (feature_map.shape[-1] - x.shape[-1])/2 

    assert diff_h.is_integer(), "Height difference is not a whole number."
    assert diff_w.is_integer(), "Width difference is not a whole number."

    cropped = feature_map[..., int(diff_h) : int(x.shape[-2] + diff_h), int(diff_w) : int(x.shape[-1] + diff_w)]

    return cropped