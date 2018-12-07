import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

from demo_forest.datasets import crop
from demo_forest.deep_learning import class_defs_2, class_defs_7

plt.ion()
################### main plotting functions ######################
def plot_data_for_tile(tile, data_bands, setup):

    """ Plots a 2 by 2 plot with data, labels and cloud mask"""
    ax = plt.subplot(2, 2, 1)
    plot_data(tile, data_bands, setup)
    plt.title(str(tile.meta_data['datetime'])[:10])

    plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
    plot_labels(tile,  setup, 'vegetation_height')
    plt.title('Vegetation height')


    plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
    plot_labels(tile,  setup, 'fractional_forest_cover')
    plt.title('Fractional forest cover')

    plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
    plot_cloud(tile, setup)
    plt.title('Cloud mask')

    plt.show()

    pass

def plot_data_for_multiple_times(tiles, data_bands, setup):
    """ Makes a plot with data from multiple tiles """
    n = len(tiles)

    for i in range(n):

        if i == 0:
            ax = plt.subplot(2, n//2, i+1)
        else:
            plt.subplot(2, n//2, i+1, sharex=ax, sharey=ax)

        plot_data(tiles[i], data_bands, setup)
        plt.axis('tight')
        plt.title(str(tiles[i].meta_data['datetime'])[:10])


    plt.show()

def plot_predicitions(tile, pred,  setup, rbg_bands =  ['B04', 'B03', 'B02']):
    if type(tile) == list:
        for i in range(len(tile)):


            plot_predicitions(tile[i], pred[i],  setup, rbg_bands )

        return

    ax = plt.subplot(2, 2, 1)
    plot_data(tile, rbg_bands, setup)
    plt.title(str(tile.meta_data['datetime'])[:10])

    plt.subplot(2, 2, 2, sharex=ax, sharey=ax)
    plot_cloud(tile, setup)
    plt.title('Cloud mask')


    plt.subplot(2, 2, 3, sharex=ax, sharey=ax)
    plot_labels(tile, setup)
    plt.title('Ground truth')

    plt.subplot(2, 2, 4, sharex=ax, sharey=ax)
    plot_prediction(pred, setup, tile)
    plt.title('Predicted')

    plt.show()

def scatter_confusion_plot(tile, pred, setup, N=10000, new_fig=True):

    if type(tile) == list:
        assert( type(pred)==list)

        n = len(tile)

        for i in range(n):
            if i == 0:
                ax = plt.subplot(2, n // 2, i + 1)
            else:
                plt.subplot(2, n // 2, i + 1, sharex=ax, sharey=ax)

            scatter_confusion_plot(tile[i], pred[i], setup, False)
        return

    ground_truth = crop(tile.get_labels([setup['label']])[0],setup)

    # Remove unlabelled pixels
    pred = pred[ground_truth >= 0]
    ground_truth = ground_truth[ground_truth >= 0]

    #Convert height to classes
    if setup['net_type'] in ['2_classes', '7_classes']:
        if setup['net_type'] == '2_classes':
            class_defs = class_defs_2
        else:
            class_defs = class_defs_7

        new_gt = (ground_truth * 0 - 100).astype('int64')  # Make new array with ignore labels
        for class_no, [from_, to_] in enumerate(class_defs):
            new_gt[(from_ <= ground_truth) & (ground_truth < to_)] = class_no
        ground_truth = new_gt

    # reduce number of points
    if len(pred) > N:
        # Select N random indexes
        inds = list(range(len(pred)))
        random.shuffle(inds)
        inds = np.array(inds[0:N])
        pred = pred[inds]
        ground_truth = ground_truth[inds]


    if setup['net_type'] in ['2_classes','7_classes']:


        label_names = [str(cd[0])+ '-' + str(cd[1]) + '' for cd in class_defs]
        #Confusion matrix
        cfm = np.zeros([len(class_defs),len(class_defs)])
        cf = confusion_matrix( ground_truth.astype('int'),pred.astype('int'),)
        vals_in_cf = np.unique(np.concatenate([pred,ground_truth]))
        for i, v in enumerate(vals_in_cf):
            for j, w in enumerate(vals_in_cf):
                cfm[int(v),int(w)] = cf[i,j]

        df_cm = pd.DataFrame(cfm, index=label_names, columns= label_names)
        sn.heatmap(df_cm, annot=True, fmt=".6g")
        ax = plt.gca()
        label = ax.set_ylabel('Ground truth', va='top' )
        label = ax.set_xlabel('Predicted', ha='right')

        total = np.sum(cfm,0)
        corrects = np.diag(cfm)
        acc = corrects/total
        print('Average class accuracy:', np.round(100*np.mean(acc[np.bitwise_not(np.isnan(acc))])))
        print('Class accuracies:', np.round(acc*100))
    else:

        #Scatter plot
        x, y = ground_truth, pred
        lims = [np.min(x), np.max(x)]
        plt.plot(lims, lims, c="k")
        plt.grid()
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        plt.scatter(x, y, c=z, s=35, edgecolor='')
        plt.xlabel("True")
        plt.ylabel("Prediction")
        plt.title(str(tile.meta_data['datetime'])[:10])
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()
        print('RMS-error:', np.sum(np.sqrt(np.mean( (x-y)**2))))



##################### utility plotting functions #################
from forest_cover_utils import color_forest_cover, color_height, color_height_classes, color_height_2_cls


# Visualize data 1
def plot_data(tile, bands, setup):
    if len(bands) != 1 and len(bands) != 3:
        print('Error: list of bands must be either have length 1 (single channel) or 3 (rgb render)')
        return

    #Get data
    bands =  tile.get_data(bands)
    bands = [crop(b, setup) for b in bands]

    #Adjust values of channels
    if len(bands)==3:
        bands = [b*4 for b in bands]
        bands = [np.clip(b,0,1) for b in bands]

    #Make one or three channel image
    bands = [np.expand_dims(b, -1) for b in bands]
    bands = np.concatenate(bands,-1)
    plt.imshow(bands)
    plt.axis('tight')
    plt.axis('off')

def plot_prediction(pred,  setup, tile_with_cloud_mask=None):
    if setup['label'] == 'fractional_forest_cover':
        label_to_color_img = color_forest_cover

    elif setup['label'] == 'vegetation_height':
        if setup['net_type'] == '2_classes':
            label_to_color_img = color_height_classes(class_defs_2)
        elif setup['net_type'] == '7_classes':
            label_to_color_img = color_height_classes( class_defs_7)
        else:
            label_to_color_img = color_height

    else:
        label_to_color_img = lambda x:x
    img = label_to_color_img(pred)[0]/255.

    if tile_with_cloud_mask is not None:
        cloud = tile_with_cloud_mask.get_cloud_mask()
        cloud = [crop(l, setup) for l in cloud][0]
        for i in range(3):
            c = img[:,:,i]
            c[cloud.squeeze()==1] = 1
            img[:,:,i] = c

    plt.imshow(img)
    plt.axis('tight')
    plt.axis('off')


def plot_labels(tile, setup, label=None):
    if label is None:
        label = setup['label']

    if label == 'fractional_forest_cover':
        label_to_color_img = color_forest_cover

    elif label == 'vegetation_height':
        if label == '2_classes':
            label_to_color_img = color_height_2_cls
        else:
            label_to_color_img = color_height

    else:
        label_to_color_img = lambda x:x

    labels = tile.get_labels([label])

    labels = [crop(l, setup) for l in labels]

    plt.imshow(label_to_color_img(labels)[0]/255.)
    plt.axis('tight')
    plt.axis('off')


def plot_cloud(tile,  setup):

    cloud = tile.get_cloud_mask()

    cloud = [crop(l, setup) for l in cloud]

    plt.imshow(cloud[0].squeeze())
    plt.axis('tight')
    plt.axis('off')
