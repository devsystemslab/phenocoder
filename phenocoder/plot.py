import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import scanpy as sc
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
from whole_mount_tumoroid.phenocoder.utils import scale_image, get_metadata
from pathlib import Path
from whole_mount_tumoroid.phenocoder.phenocode import encode_nuclei_patches
from skimage.util import montage
from skimage import io
from math import cos, sin, radians, sqrt
from skimage.filters.rank import median
from skimage import morphology
from skimage.exposure import rescale_intensity
from skimage.util import dtype_limits


def plot_intensities_vs_z(adata: ad.AnnData) -> None:
    """
    Plot intensities vs z
    :param adata:
    :return:
    """
    n_features = adata.var_names.shape[0]
    n_row = 2
    fig, axes = plt.subplots(n_row, int(np.ceil(n_features / n_row)), figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        sc.pl.scatter(adata, x='z', y=adata.var_names[i], ax=ax, show=False)
    # set title
    fig.suptitle('Intensity corrected vs z-axis')
    plt.show()


def plot_plate_3d(adata: ad.AnnData, plate: str, dir_screen: str,
                  leiden_colors=None) -> None:
    """
    Plot plate overview 3d
    :param adata:
    :param plate:
    :param dir_screen:
    :param leiden_colors:
    :return:
    """
    file_plate_layout = Path(dir_screen, 'plate_layout.csv')
    # filter for plate
    adata = adata[adata.obs['plate_id'] == plate]
    adata.obs.index = adata.obs.index.astype('str')
    # create colormap
    if leiden_colors is None:
        leiden_colors = adata.uns['leiden_colors']
    cmap = ListedColormap(leiden_colors)
    # load plate layout
    df_plate_layout = pd.read_csv(file_plate_layout,
                                  dtype={'well': str, 'plate': str})
    wells = df_plate_layout['well'].unique().tolist()
    # plot centroid-0 and centroid-1 for each well
    fig = plt.figure(figsize=(36, 24))
    for i, well in enumerate(wells):
        adata_tmp = adata[adata.obs['well_id'] == wells[i]]
        center_centroid_0 = adata_tmp.obs['centroid-0'].mean()
        center_centroid_1 = adata_tmp.obs['centroid-1'].mean()
        # set to str index
        adata_tmp.obs.index = adata_tmp.obs.index.astype('str')
        # filter points that are smaller
        adata_tmp = adata_tmp[
            ~((adata_tmp.obs['centroid-0'] > center_centroid_0) & (adata_tmp.obs['centroid-1'] < center_centroid_1))]
        ax = fig.add_subplot(16, 24, i + 1, projection='3d')
        if i == 0:
            plt.legend([mpatches.Patch(color=cmap(b)) for b in np.unique(adata.obs['leiden'].astype(int))],
                       np.unique(adata.obs['leiden'].astype(str)), loc='upper left', bbox_to_anchor=(-2, 1))
        if adata_tmp.shape[0] > 0:
            ax.scatter(adata_tmp.obs['centroid-0'], adata_tmp.obs['centroid-1'], adata_tmp.obs['z'],
                       c=adata_tmp.obs['leiden'].astype(int), s=1, alpha=0.75, cmap=cmap)
        # no axis labels
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.get_zaxis().set_ticklabels([])
        # invert y-axis
    plt.show()


def plot_clustermap(adata, cluster_key='leiden'):
    """
    Plot clustermap
    :param adata:
    :param cluster_key:
    :return:
    """
    # df with X and leiden labels for each cell
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df[cluster_key] = adata.obs[cluster_key].values.astype(str)
    df.set_index(cluster_key, inplace=True)
    # sns clustermap
    df_avg = df.groupby(cluster_key, observed=False).median()
    # map cmap to leiden_colors
    if f'{cluster_key}_colors' in adata.uns:
        sns.clustermap(df_avg, cmap='vlag', z_score=0, row_colors=adata.uns[f'{cluster_key}_colors'], figsize=(5, 5))
    else:
        sns.clustermap(df_avg, cmap='vlag', z_score=0, figsize=(5, 5))
    plt.show()


def plot_results(adata_nuc, adata_spatial, plates, adata_org=None):
    # plot intensities vs z
    plot_intensities_vs_z(adata_nuc)
    plot_intensities_vs_z(adata_spatial)

    # plot umaps
    sc.pl.umap(adata_nuc, color='leiden')
    plt.show()
    sc.pl.umap(adata_nuc, color=adata_nuc.var_names)
    plt.show()
    sc.pl.umap(adata_spatial, color='leiden')
    plt.show()
    sc.pl.umap(adata_spatial, color=adata_spatial.var_names)
    plt.show()

    # plot clustermaps
    plot_clustermap(adata_nuc)
    plot_clustermap(adata_spatial)
    # plot plate overview 3d
    for plate in plates:
        plot_plate_3d(adata_spatial, plate)
        plot_plate_3d(adata_nuc, plate)
    if adata_org is not None:
        # organoid embeddings
        plt_features = ['conc', 'timepoint', 'plate', 'compound']
        sc.pl.umap(adata_org, color=plt_features)
        plt.show()
        sc.pl.pca(adata_org, color=plt_features)
        plt.show()


def plot_organoid(adata, well: str, plate: str, cut_open=True, clusters=None):
    """
    Plot organoid
    :param adata:
    :param well:
    :param plate:
    :param cut_open:
    :param clusters:
    :return:
    """

    adata = adata[adata.obs['well_id'] == well]
    adata = adata[adata.obs['plate_id'] == plate]
    if clusters is not None:
        adata = adata[adata.obs['leiden'].isin(clusters)]
    adata.obs.index = adata.obs.index.astype('str')

    if 'leiden_colors' not in adata.uns:
        cmap = ListedColormap(sns.color_palette('tab20', n_colors=adata.obs['leiden'].nunique()))
    else:
        cmap = ListedColormap(adata.uns['leiden_colors'])
    fig = plt.figure(figsize=(6, 18))
    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(adata.obs['centroid-0'], adata.obs['centroid-1'],
               c=adata.obs['leiden'].astype(int), s=10, cmap=cmap)
    # same aspects in x and y
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(3, 1, 2, projection='3d')
    ax.scatter(adata.obs['centroid-0'], adata.obs['centroid-1'], adata.obs['z'],
               c=adata.obs['leiden'].astype(int), s=10, cmap=cmap, alpha=0.75)
    if cut_open:
        center_centroid_0 = adata.obs['centroid-0'].mean()
        center_centroid_1 = adata.obs['centroid-1'].mean()
        adata = adata[
            ~((adata.obs['centroid-0'] > center_centroid_0) & (
                    adata.obs['centroid-1'] < center_centroid_1))]
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    ax.scatter(adata.obs['centroid-0'], adata.obs['centroid-1'], adata.obs['z'],
               c=adata.obs['leiden'].astype(int), s=10, cmap=cmap, alpha=0.75)

    plt.legend([mpatches.Patch(color=cmap(b)) for b in np.unique(adata.obs['leiden'].astype(int))],
               np.unique(adata.obs['leiden'].astype(str)), loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_organoid_with_image(adata, well: str, plate: str, dir_screen: str, cycle: str,
                             z: int = None,
                             apply_median:tuple[str]=None,
                             median_size=5,
                             lut_dict=None,
                             save: bool = False,
                             plot: bool = True):
    """
    Plot organoid with image
    :param adata:
    :param well:
    :param plate:
    :param cycle:
    :param z:
    :param lut_dict:
    :param apply_median:
    :param median_size:
    :param dir_screen:
    :param save:
    :param plot:
    :return:
    """

    if save:
        dir_output = Path(dir_screen, 'example_overlays', f'{plate}-{cycle}', well)
        dir_output.mkdir(parents=True, exist_ok=True)

    if lut_dict is None:
        if cycle == '03':
            lut_dict = {'01': (1, 99),
                        '02': (1, 99),
                        '03': (95, 99),
                        '04': (1, 99)}
        else:
            lut_dict = {'01': (1, 99),
                        '02': (1, 99),
                        '03': (1, 99),
                        '04': (1, 99)}

    if apply_median is None:
        if cycle == '03':
            apply_median = ('02', '03')
        if cycle == '01':
            apply_median = ('02','03')

    # filter adata for well and plate
    adata = adata[adata.obs['well_id'] == well]
    adata = adata[adata.obs['plate_id'] == plate]

    df_images = get_metadata(Path(dir_screen, plate, f'{plate}-{cycle}','TIF_OVR_BG'))
    df_images = df_images[df_images['well_id'] == well]

    df_segmentation = get_metadata(Path(dir_screen, plate, f'{plate}-{cycle}', 'SEG_TIF_OVR_BG'))
    df_segmentation = df_segmentation[df_segmentation['well_id'] == well]

    if z is not None:
        adata = adata[adata.obs['z'] == z]
        df_segmentation = df_segmentation[df_segmentation['z_stack_id'] == str(z)]
        df_images = df_images[df_images['z_stack_id'] == str(z)]

    # arrange by channel_id
    df_images = df_images.sort_values(by=['z_stack_id','channel_id'])
    df_segmentation = df_segmentation.sort_values(by=['z_stack_id'])

    imgs = []
    for channel in df_images['channel_id'].unique():
        df_images_channel = df_images[df_images['channel_id'] == channel]
        imgs_channel = np.asarray([io.imread(Path(dir_screen, plate, f'{plate}-{cycle}', 'TIF_OVR_BG', file)) for file in tqdm(df_images_channel['file'], desc=f'Loading channel {channel}', total=len(df_images_channel))])

        imgs_channel = imgs_channel.max(axis=0)
        if apply_median is not None:
            if channel in apply_median:
                print(f'Applying median: {channel}')
                imgs_channel = median(imgs_channel, morphology.disk(median_size))
        range_lut = (int(np.percentile(imgs_channel, lut_dict[channel][0])),
                     int(np.percentile(imgs_channel, lut_dict[channel][1])))
        imgs_channel = rescale_intensity(imgs_channel,in_range=range_lut, out_range=(0, 1.0))

        imgs.append(imgs_channel)
    imgs = np.asarray(imgs)
    for channel in range(imgs.shape[0]):
        if plot:
            fig = plt.figure(figsize=(20, 20))
            io.imshow(imgs[channel])
            plt.show()
            plt.close('all')
        if save:
            suffix = ['max' if z is None else f'z_{z}'][0]
            io.imsave(Path(dir_output, f'{channel}-{suffix}.png'), rescale_intensity(imgs[channel], out_range=(0, 255)).astype(np.uint8))
    # move color channel to last axis
    imgs = np.moveaxis(imgs, 0, -1)
    img = imgs[..., 1:]
    img_2 = imgs[..., 0]
    # duplicate img2 3 times in axis
    img_2 = np.repeat(img_2[:, :, np.newaxis], 3, axis=2)
    img = (img + img_2) / 2
    img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    # hue rotation
    rotator = RGBRotate()
    rotator.set_hue_rotation(49)
    img = rotator.apply_to_image(img)
    img = scale_image(img, range=(0,255)).astype(np.uint8)
    if plot:
        fig = plt.figure(figsize=(20,20))
        io.imshow(img)
        plt.show()
        plt.close('all')
    if save:
        suffix = ['max' if z is None else f'z_{z}'][0]
        io.imsave(Path(dir_output, f'overlay-{suffix}.png'), img)

def clamp(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v + 0.5)

class RGBRotate(object):
    def __init__(self):
        self.matrix = [[1,0,0],[0,1,0],[0,0,1]]

    def set_hue_rotation(self, degrees):
        cosA = cos(radians(degrees))
        sinA = sin(radians(degrees))
        self.matrix[0][0] = cosA + (1.0 - cosA) / 3.0
        self.matrix[0][1] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[0][2] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[1][0] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[1][1] = cosA + 1./3.*(1.0 - cosA)
        self.matrix[1][2] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[2][0] = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA
        self.matrix[2][1] = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA
        self.matrix[2][2] = cosA + 1./3. * (1.0 - cosA)

    def apply(self, r, g, b):
        rx = r * self.matrix[0][0] + g * self.matrix[0][1] + b * self.matrix[0][2]
        gx = r * self.matrix[1][0] + g * self.matrix[1][1] + b * self.matrix[1][2]
        bx = r * self.matrix[2][0] + g * self.matrix[2][1] + b * self.matrix[2][2]
        return clamp(rx), clamp(gx), clamp(bx)

    def apply_to_image(self, img):
        img = img.astype(np.int16)
        if img.shape[-1] == 3:
            img_rx =  img[...,0] * self.matrix[0][0] + img[...,1] * self.matrix[0][1] + img[...,2] * self.matrix[0][2]
            img_gx = img[...,0] * self.matrix[1][0] + img[...,1] * self.matrix[1][1] + img[...,2] * self.matrix[1][2]
            img_bx = img[...,0] * self.matrix[2][0] + img[...,1] * self.matrix[2][1] + img[...,2] * self.matrix[2][2]
            img = np.asarray([img_rx, img_gx, img_bx])
            img = np.moveaxis(img, 0, -1)
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img


def show_rotation(rotation):
    """
    Generate white, black, red, green, blue striped image and apply hue rotation to it and plot result.
    :param rotation:
    :return:
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:20, :, :] = [255, 255, 255]
    img[20:40, :, :] = [0, 0, 0]
    img[40:60, :, :] = [255, 0, 0]
    img[60:80, :, :] = [0, 255, 0]
    img[80:100, :, :] = [0, 0, 255]
    img_init = img.copy()
    rotator = RGBRotate()
    rotator.set_hue_rotation(rotation)
    img = np.asarray([rotator.apply(r, g, b) for r, g, b in img.reshape(-1, 3)]).reshape(img.shape).astype(np.uint8)
    io.imshow(np.hstack((img_init, img)))
    plt.show()

def add_scalebar(img: np.ndarray, width:int, height:int, x:int, y:int) -> np.ndarray:
    """
    Add scale bar to image
    :param img:
    :param width:
    :param height:
    :param x:
    :param y:
    :return:
    """
    img = img.copy()
    # get dtype max
    max_val = dtype_limits(img)[1]
    img[y:y+height,x:x+width,:] = max_val
    return img

def generate_examples(adata_org: ad.AnnData,
                      cycles: tuple = ('01', '03'),
                      lut_dict=None,
                      n: int = 3,
                      n_down_sampling: int = None,
                      input_type: str = 'TIF_MIP_OVR_BG',
                      scale_bar: int = None,
                      dir_screen: str = '/pstore/data/ihb-tumoroidscreen/data/processed/tumoroidscreen'):
    """
    Generate example images
    :param adata_org:
    :param cycles:
    :param lut_dict:
    :param n:
    :param n_down_sampling:
    :param input_type:
    :param scale_bar:
    :param dir_screen:
    :return:
    """
    if lut_dict is None:
        lut_dict = {'01': [(1, 99), (1, 99), (1, 99), (1, 99)],
                    '03': [(5, 95), (5, 95), (95, 99), (5, 95)]}
    dir_output = Path(dir_screen, 'example_images')
    dir_output.mkdir(exist_ok=True, parents=True)
    if n is not None:
        df_iter = adata_org.obs.groupby('leiden', observed=False).apply(lambda x: x.sample(n)).reset_index(drop=True)
    else:
        df_iter = adata_org.obs.copy()
    processed_files = []
    for i, row in tqdm(df_iter.iterrows(), desc='Generating example images', total=df_iter.shape[0]):
        for cycle in cycles:
            dir_images = os.path.join(dir_screen, row['plate_id'], row['plate_id'] + f'-{cycle}', input_type)
            files = [f for f in os.listdir(dir_images) if
                     f.endswith('.tif')]
            files_well = sorted([f for f in files if '_' + row['well_id'] + '_' in f], reverse=True)
            file_out = row['well_id'] + '_' + row['plate_id'] + '_c' + row['leiden'] + f'_cycle_{cycle}.png'
            file_out = os.path.join(dir_output, file_out)
            imgs = []
            if len(files_well) == 0:
                imgs = np.zeros((4,3814,3814))
            else:
                for channel, file in enumerate(files_well):
                   img = io.imread(os.path.join(dir_images, file))
                   range_lut = (np.percentile(img,lut_dict[cycle][channel][0]),
                                np.percentile(img,lut_dict[cycle][channel][1]))
                   img = rescale_intensity(img,
                                           in_range=range_lut,
                                           out_range=(0, 1.0))
                   imgs.append(img)
                imgs = np.asarray(imgs)
            # subsample pixels
            if n_down_sampling is not None:
                imgs = np.asarray([img[::n_down_sampling, ::n_down_sampling] for img in imgs])
            # move color channel to last axis
            imgs = np.moveaxis(imgs, 0, -1)
            img = imgs[..., 1:]
            img_2 = imgs[..., 0]
            # duplicate img2 3 times in axis
            img_2 = np.repeat(img_2[:, :, np.newaxis], 3, axis=2)
            img = (img + img_2) / 2
            img = rescale_intensity(img,  out_range=(0,255)).astype(np.uint8)
            # hue rotation
            rotator = RGBRotate()
            rotator.set_hue_rotation(49)
            img = np.asarray([rotator.apply(r, g, b) for r, g, b in img.reshape(-1, 3)]).reshape(img.shape).astype(np.uint8)
            img = scale_image(img, range=(0,255)).astype(np.uint8)
            # add scale bar
            if scale_bar is not None:
                height = 50
                offset_x, offset_y = 100, 100
                if n_down_sampling is not None:
                    width = int(scale_bar/n_down_sampling)
                    height = int(height/n_down_sampling)
                    offset_x = int(offset_x/n_down_sampling)
                    offset_y = int(offset_y / n_down_sampling)
                    img = add_scalebar(img,
                                       width = width,
                                       height = height,
                                       x = img.shape[1]- offset_x - width,
                                       y = img.shape[1] - offset_y - height)
                else:
                    img = add_scalebar(img, scale_bar, height, offset_x, offset_y)
            # save as png
            io.imsave(file_out, img)
            processed_files.append(file_out)
    if cycles == ('01', '03'):
        df_iter = df_iter.assign(file_cycle_1=[file for file in processed_files if file.endswith('cycle_01.png')],
                                 file_cycle_3=[file for file in processed_files if file.endswith('cycle_03.png')])
    else:
        df_iter = df_iter.assign(file_cycle_1=[file for file in processed_files if file.endswith('cycle_01.png')])
    df_iter.to_csv(os.path.join(dir_output, 'example_images.csv'), index=False)

def generate_example_patch_montage(adata, n_patches, params, cycle='01'):
    """
    Generate example images for nuclei patches and montage them
    :param adata:
    :param n_patches:
    :param params:
    :param cycle:
    :return:
    """
    dir_screen = params['dir_screen']
    model_dict = params['phenocoder']['models']
    # setup dir_output
    dir_output = Path(dir_screen, 'example_patches',f'cycle-{cycle}')
    dir_output.mkdir(parents=True, exist_ok=True)
    # group adata.obs by leiden and sample n_patches per group
    df_iter = adata.obs.groupby('leiden', observed=False).apply(lambda x: x.sample(n_patches))
    df_iter = df_iter.drop(columns='leiden').reset_index()
    clusters = df_iter['leiden'].unique().tolist()
    montages = []
    for cluster in clusters:
        results_cluster_patches = []
        results_cluster_df = []
        df_cluster = df_iter[df_iter['leiden'] == cluster]
        plates = df_cluster['plate_id'].unique().tolist()
        for plate in plates:
            df_plate = df_cluster[(df_cluster['plate_id'] == plate)]
            if df_plate.empty:
                continue
            # restore original labels
            df_plate['label'] = df_plate['label'].apply(lambda x: x.split('_')[0])
            well_ids = df_plate['well_id'].unique().tolist()
            if model_dict['source']['cycle'] == cycle:
                dir_model_cycle = model_dict['source']['file']
            elif model_dict['target']['cycle'] == cycle:
                dir_model_cycle = model_dict['target']['file']
            else:
                raise ValueError(f'Cycle {cycle} not found in model_dict')
            patches_init, df = map(list,zip(*encode_nuclei_patches(well_ids=well_ids,
                                  plate=plate,
                                  cycle=cycle,
                                  dir_screen=dir_screen,
                                  df_labels=df_plate,
                                  use_registered=False,
                                  dir_model=Path(params['phenocoder']['dir_models'], dir_model_cycle),
                                  return_only_patches=True)))
            patches = []
            for i, (patch, df_well) in enumerate(zip(patches_init, df)):
                df_well = df_well.drop(columns=['well_id', 'plate_id'])
                if df_well.shape[0] > 1:
                    df_agg = df_well.groupby('label').mean().reset_index()
                    if df_agg.shape[0] > 1:
                        patch_tmp = np.zeros((df_agg.shape[0], patch.shape[1], patch.shape[2], patch.shape[3]))
                        for n, label in enumerate(df_agg['label']):
                            idx = df_well['label'] == label
                            patch_idx = patch[idx]
                            if patch_idx.shape[0] > 1:
                                patch_tmp[n] = np.max(patch_idx, axis=0)
                            else:
                                patch_tmp[n] = np.squeeze(patch_idx, axis=0)
                        patch = patch_tmp.copy()
                    else:
                        patch = np.max(patch, axis=0)
                    df_well = df_agg.copy()
                else:
                    patch = np.squeeze(patch, axis=0)
                df[i] = df_well.copy()
                if len(patch.shape) > 3:
                    for n in range(patch.shape[0]):
                        patches.append(patch[n])
                else:
                   patches.append(patch)
            results_cluster_patches.extend(patches)
            results_cluster_df.extend(df)
        imgs = np.asarray(results_cluster_patches)
        img = imgs[..., 1:]
        img_2 = imgs[..., 0]
        img_2 = np.repeat(img_2[..., np.newaxis], 3, axis=-1)
        img = (img + img_2) / 2
        img_montage = montage(img, channel_axis=-1)
        montages.append(img_montage)
        for channel in range(img_montage.shape[-1]):
            img_montage[..., channel] = scale_image(img_montage[..., channel], range=(0, 255), percentile=1)
        img_montage = img_montage.astype(np.uint8)
        # save montage
        io.imsave(Path(dir_output, f'cluster_{cluster}_montage.png'), img_montage)
    montages = np.asarray(montages)
    montage_all = montage(montages, grid_shape=(1, len(clusters)), channel_axis=-1)
    for channel in range(montage_all.shape[-1]):
        montage_all[..., channel] = scale_image(montage_all[..., channel], range=(0, 255), percentile=1)
    io.imsave(Path(dir_output, 'montage_all.png'),  montage_all.astype(np.uint8))


def plot_history(model, show=True, return_fig=False):
    if model.history is None:
        raise ValueError('No history available')

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(model.history.history['loss'], label='loss')
    ax[0].plot(model.history.history['val_loss'], label='val_loss')
    ax[0].set_title('Total Loss')
    ax[0].legend()

    ax[1].plt(model.history.history['kl_loss'], label='kl_loss')
    ax[1].plot(model.history.history['val_kl_loss'], label='val_kl_loss')
    ax[1].set_title('KL Loss')
    ax[1].legend()

    ax[2].plot(model.history.history['reconstruction_loss'], label='reconstruction_loss')
    ax[2].plot(model.history.history['val_reconstruction_loss'], label='val_reconstruction_loss')
    ax[2].set_title('Reconstruction Loss')
    ax[2].legend()
    if show:
        plt.show()
    if return_fig:
        return fig

def plot_organoid_test(adata, res=1, layer=None):
    """
    Plot organoid in 3D
    :param adata:
    :param res:
    :return:
    """
    # pca
    sc.tl.pca(adata, n_comps=16, layer=layer)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=res)
    sc.pl.umap(adata, color='leiden')

    fig = plt.figure(figsize=(24, 24))
    well_ids = adata.obs['well_id'].unique()
    cmap = ListedColormap(adata.uns['leiden_colors'])
    for i,well in enumerate(well_ids):
        adata_well = adata[adata.obs['well_id'] == well]
        ax = fig.add_subplot(2,len(well_ids),i+1, projection='3d')
        ax.scatter(adata_well.obs['x'], adata_well.obs['y'], adata_well.obs['z'],
                   c=adata_well.obs['leiden'].astype(int), s=20, cmap=cmap, alpha=0.75)
        ax = fig.add_subplot(2, len(well_ids), i + 1 + len(well_ids))
        sc.pl.scatter(adata_well, color='leiden', x='x', y='y', size=40, show=False, ax=ax)

    plt.show()