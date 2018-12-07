import numpy as np

##################### utility deep learning functions #################
from pytorch_utils import torch_to_np, np_to_torch, gpu_no_of_var


def apply_net_to_large_data(data, net, patch_size, patch_overlap, apply_classifier=True):
    """
    Chops up a large image in smaller patches and run each patch through a segmentation network. The output is stitched
    together from the predicted patches.
    :param data: The large image (np.array 2D (single channel) or 3D (multiple channels)
    :param net: A pytorch segmentation model (input size must be equal to output size)
    :param patch_size: Size of patches ([int,int])
    :param patch_overlap: How much overlap there should be between patches ([int,int])
    :param apply_classifier: Apply argmax across of ouput channels (bool)
    :return: Predictions for large image (np.array 2D (single channel) or 3D (multiple channels)
    """

    #Functions to convert between B x C x H x W format and W x H x C format
    def hwc_to_bchw(x):
        return np.expand_dims(np.moveaxis(x,-1,0),0)

    def bcwh_to_hwc(x):
        return np.moveaxis(x.squeeze(0),0,-1)

    if type(patch_size) == int:
        patch_size = [patch_size, patch_size]

    if len(data.shape)==2:
        data = np.expand_dims(data,-1)

    #Add padding to avoid trouble when removing the overlap later
    data = np.pad(data, [[patch_overlap[0], patch_overlap[0]], [patch_overlap[1], patch_overlap[1]], [0, 0]], 'constant')

    # Loop through patches identified by upper-left pixel
    upper_left_x0 = np.arange(0, data.shape[0] - patch_overlap[0], patch_size[0] - patch_overlap[0] * 2)
    upper_left_x1 = np.arange(0, data.shape[1] - patch_overlap[1], patch_size[1] - patch_overlap[1] * 2)

    predictions = []
    for x0 in upper_left_x0:
        for x1 in upper_left_x1:
            #Cut out a small patch of the data
            data_patch = data[x0:x0 + patch_size[0], x1:x1 + patch_size[1], :]

            # Pad with zeros if we are at the edges
            pad_val_0 = patch_size[0] - data_patch.shape[0]
            pad_val_1 = patch_size[1] - data_patch.shape[1]

            if pad_val_0 > 0:
                data_patch = np.pad(data_patch, [ [0, pad_val_0], [0, 0], [0, 0]], 'constant')

            if pad_val_1 > 0:
                data_patch = np.pad(data_patch, [[0, 0], [0, pad_val_1], [0, 0]], 'constant')

            # Run it through model
            out_patch = net(np_to_torch(hwc_to_bchw(data_patch), gpu_no_of_var(net)).float())
            out_patch = bcwh_to_hwc(torch_to_np(out_patch))

            #Argmax for classifications
            if out_patch.shape[2]>1 and apply_classifier:
                out_patch = np.argmax(out_patch, axis=2)
                out_patch = np.expand_dims(out_patch, -1)

            # Make output array (We do this here since it will then be agnostic to the number of output channels)
            if len(predictions)==0:
                predictions = np.concatenate([data[:-(patch_overlap[0] * 2), :-(patch_overlap[1] * 2), 0:1] * 0] * out_patch.shape[2], -1)

            #Remove eventual padding related to edges
            out_patch = out_patch[0:patch_size[0] - pad_val_0, 0:patch_size[1] - pad_val_1, :]

            # Remove eventual padding related to overlap between data_patches
            out_patch = out_patch[patch_overlap[0]:-patch_overlap[0], patch_overlap[1]:-patch_overlap[1], :]

            # Insert output_patch in out array
            predictions[x0:x0 + out_patch.shape[0], x1:x1 + out_patch.shape[1], :] = out_patch

    return predictions