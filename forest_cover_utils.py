import matplotlib.cm
import numpy as np
from sentinel_dataset import make_visualize_function


def plot_training_sample(input_dict):
    from dlt.utils.plotting import setup_matplotlib
    plt = setup_matplotlib()
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(make_visualize_function(input_dict['data_bands'])(input_dict['data']))
    plt.title('data')
    plt.subplot(1, 3, 2)
    plt.imshow(input_dict['target'][:,:,0])
    plt.title('target')
    plt.subplot(1, 3, 3)
    plt.imshow(input_dict['cloud'][:,:,0])
    plt.title('cloud')
    plt.show()
    return input_dict


def color_height_2_cls(height):
    """ Color image as in
    Multi-sensor forest vegetation height mapping methods for Tanzania
    https://www.tandfonline.com/doi/full/10.1080/22797254.2018.1461533"""

    color_table = [
        [[0,2], [250, 196, 57]],
        [[5,100], [45, 156, 76]],
    ]

    if type(height) not in [list, tuple]:
        height = [height]

    out = []
    for h in height:
        new_im = np.ones(list(h.shape) + [3])*255

        inds = np.meshgrid(np.linspace(0, h.shape[0] - 1, h.shape[0], dtype='int64'),
                           np.linspace(0, h.shape[1] - 1, h.shape[1], dtype='int64'), indexing='ij')

        for i in range(len(color_table)):
            try:
                h = h.squeeze(2)
            except:
                pass
            ind = np.bitwise_and(h >= color_table[i][0][0], h <= color_table[i][0][1])
            y = inds[0][ind]
            x = inds[1][ind]
            new_im[y, x, :] = color_table[i][1]

        out.append(new_im)

    return out

def color_height(height):
    """ Color image as in
    Multi-sensor forest vegetation height mapping methods for Tanzania
    https://www.tandfonline.com/doi/full/10.1080/22797254.2018.1461533"""

    color_table = [
        [0, [250, 196, 57]],
        [1, [246, 237, 64]],
        [2, [87, 184, 78]],
        [5, [45, 156, 76]],
        [10, [34, 100, 55]],
        [20, [35, 73, 35]],
        [40, [25, 47, 24]],
    ]



    new_im = np.ones(list(height.shape[0:2]) + [3])*255

    inds = np.meshgrid(np.linspace(0, height.shape[0] - 1, height.shape[0], dtype='int64'),
                       np.linspace(0, height.shape[1] - 1, height.shape[1], dtype='int64'), indexing='ij')

    for i in range(len(color_table)):
        try:
            height = height.squeeze(2)
        except:
            pass
        ind = height >= color_table[i][0]
        y = inds[0][ind]
        x = inds[1][ind]
        new_im[y, x, :] = color_table[i][1]


    return new_im


def color_height_classes(class_defs):
    def func(height):
        """ Color image as in
        Multi-sensor forest vegetation height mapping methods for Tanzania
        https://www.tandfonline.com/doi/full/10.1080/22797254.2018.1461533"""


        color_table = [
            [0, [250, 196, 57]],
            [1, [246, 237, 64]],
            [2, [87, 184, 78]],
            [5, [45, 156, 76]],
            [10, [34, 100, 55]],
            [20, [35, 73, 35]],
            [40, [25, 47, 24]],
            [4000000000000, [25, 47, 24]],
        ]

        #Make a color-table for the class-defs given
        color_table_for_class = []
        for cd in class_defs:
            for i in range(len(color_table)-1):
                if cd[0] >= color_table[i][0] and cd[0] < color_table[i+1][0]:
                    color_table_for_class.append(color_table[i][1])

        if len(height.shape) > 2:
            height = height.squeeze(2)


        #New image
        new_im = np.ones(list(height.shape[0:2]) + [3])*255 #Intialize with white color

        inds = np.meshgrid(np.linspace(0, height.shape[0] - 1, height.shape[0], dtype='int64'),
                           np.linspace(0, height.shape[1] - 1, height.shape[1], dtype='int64'), indexing='ij')

        #Loop through classes and add color to pixels with the given class
        for i in range(len(color_table_for_class)):

            ind = height==i
            y = inds[0][ind]
            x = inds[1][ind]
            new_im[y, x, :] = color_table_for_class[i]

        return new_im
    return func

def color_tree_cover(cover):
    """ Clips pixelvalues in  tree-cover to [0, 100] and makes sure imshow-scales between these two"""


    #Set ignores to nan (white)
    cmap = matplotlib.cm.get_cmap('viridis')
    try:
        out = cover[:, :, 0]
    except:
        pass
    ignores = np.bitwise_or(out == -1, out == -100)
    out[out < 0] = 0
    out[out > 100] = 100
    out /=100
    out[0, 0] = 0
    out[0, 1] = 1
    out = cmap(out)[:, :, 0:3]
    for i in range(3):
        cc = out[:, :, i]
        cc[ignores] = 1
        out[:, :, i] = cc

    return out * 255


def color_tree_height_plus_cover(output):
    if type(output) == list:
        output = [np.expand_dims(o,-1) for o in output]
        output = np.concatenate(output,-1)

    return color_height(output[:,:,0])[0], color_tree_cover(output[:,:,1])[0]