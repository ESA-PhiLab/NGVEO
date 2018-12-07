import numpy as np

def make_batch(ds, win_size, batch_size, mask_clouds=True):

    data = []
    target = []

    for i in range(batch_size):
        sample = ds[ np.random.randint(len(ds)) ]
        x,y = sample['coord']
        x, y = sample['coord']

        #Get data channels
        d = [band[y+(-win_size[0]//2): y+(win_size[0]//2), x+(-win_size[1]//2): x+(win_size[1]//2)] for band in sample['data']]
        d = [np.expand_dims(d_,0) for d_ in d]
        d = np.concatenate(d,0)
        data.append(np.expand_dims(d,0))

        #Get target channel
        d = [band[y + (-win_size[0] // 2): y + (win_size[0] // 2), x + (-win_size[1] // 2): x + (win_size[1] // 2)] for band in sample['target']]
        d = [np.expand_dims(d_, 0) for d_ in d]
        d = np.concatenate(d, 0)

        if mask_clouds:
            missing_data = sample['missing_data_mask'][0][y + (-win_size[0] // 2): y + (win_size[0] // 2), x + (-win_size[1] // 2): x + (win_size[1] // 2)]
            missing_data = np.expand_dims(missing_data, 0).squeeze(-1)
            d[missing_data==True] = -100

        target.append(np.expand_dims(d, 0))

    data = np.concatenate(data, 0)
    target = np.concatenate(target, 0)

    #Remove regions where the data is undefined
    target[np.any(np.isnan(data),1,keepdims=True)] = -100
    target[np.any(np.isnan(target), 1, keepdims=True)] = -100
    data[np.isnan(data)] = 0

    return data, target