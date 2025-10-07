import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from phenocoder.utils import scale_image
from pathlib import Path
from phenocoder.phenocode import encode_nuclei_patches
from skimage.util import montage
from skimage import io
from skimage.util import dtype_limits


def plot_plate_3d(
    adata: ad.AnnData, plate: str, dir_screen: str, leiden_colors=None
) -> None:
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
    df_plate_layout = pd.read_csv(file_plate_layout, dtype={'well': str, 'plate': str})
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
            ~(
                (adata_tmp.obs['centroid-0'] > center_centroid_0)
                & (adata_tmp.obs['centroid-1'] < center_centroid_1)
            )
        ]
        ax = fig.add_subplot(16, 24, i + 1, projection='3d')
        if i == 0:
            plt.legend(
                [
                    mpatches.Patch(color=cmap(b))
                    for b in np.unique(adata.obs['leiden'].astype(int))
                ],
                np.unique(adata.obs['leiden'].astype(str)),
                loc='upper left',
                bbox_to_anchor=(-2, 1),
            )
        if adata_tmp.shape[0] > 0:
            ax.scatter(
                adata_tmp.obs['centroid-0'],
                adata_tmp.obs['centroid-1'],
                adata_tmp.obs['z'],
                c=adata_tmp.obs['leiden'].astype(int),
                s=1,
                alpha=0.75,
                cmap=cmap,
            )
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
        sns.clustermap(
            df_avg,
            cmap='vlag',
            z_score=0,
            row_colors=adata.uns[f'{cluster_key}_colors'],
            figsize=(5, 5),
        )
    else:
        sns.clustermap(df_avg, cmap='vlag', z_score=0, figsize=(5, 5))
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
        cmap = ListedColormap(
            sns.color_palette('tab20', n_colors=adata.obs['leiden'].nunique())
        )
    else:
        cmap = ListedColormap(adata.uns['leiden_colors'])
    fig = plt.figure(figsize=(6, 18))
    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(
        adata.obs['centroid-0'],
        adata.obs['centroid-1'],
        c=adata.obs['leiden'].astype(int),
        s=10,
        cmap=cmap,
    )
    # same aspects in x and y
    ax.set_aspect('equal', 'box')
    ax = fig.add_subplot(3, 1, 2, projection='3d')
    ax.scatter(
        adata.obs['centroid-0'],
        adata.obs['centroid-1'],
        adata.obs['z'],
        c=adata.obs['leiden'].astype(int),
        s=10,
        cmap=cmap,
        alpha=0.75,
    )
    if cut_open:
        center_centroid_0 = adata.obs['centroid-0'].mean()
        center_centroid_1 = adata.obs['centroid-1'].mean()
        adata = adata[
            ~(
                (adata.obs['centroid-0'] > center_centroid_0)
                & (adata.obs['centroid-1'] < center_centroid_1)
            )
        ]
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    ax.scatter(
        adata.obs['centroid-0'],
        adata.obs['centroid-1'],
        adata.obs['z'],
        c=adata.obs['leiden'].astype(int),
        s=10,
        cmap=cmap,
        alpha=0.75,
    )

    plt.legend(
        [
            mpatches.Patch(color=cmap(b))
            for b in np.unique(adata.obs['leiden'].astype(int))
        ],
        np.unique(adata.obs['leiden'].astype(str)),
        loc='upper left',
        bbox_to_anchor=(1, 1),
    )
    plt.show()


def add_scalebar(
    img: np.ndarray, width: int, height: int, x: int, y: int
) -> np.ndarray:
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
    img[y : y + height, x : x + width, :] = max_val
    return img


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
    dir_output = Path(dir_screen, 'example_patches', f'cycle-{cycle}')
    dir_output.mkdir(parents=True, exist_ok=True)
    # group adata.obs by leiden and sample n_patches per group
    df_iter = adata.obs.groupby('leiden', observed=False).apply(
        lambda x: x.sample(n_patches)
    )
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
            patches_init, df = map(
                list,
                zip(
                    *encode_nuclei_patches(
                        well_ids=well_ids,
                        plate=plate,
                        cycle=cycle,
                        dir_screen=dir_screen,
                        df_labels=df_plate,
                        use_registered=False,
                        dir_model=Path(
                            params['phenocoder']['dir_models'], dir_model_cycle
                        ),
                        return_only_patches=True,
                    )
                ),
            )
            patches = []
            for i, (patch, df_well) in enumerate(zip(patches_init, df)):
                df_well = df_well.drop(columns=['well_id', 'plate_id'])
                if df_well.shape[0] > 1:
                    df_agg = df_well.groupby('label').mean().reset_index()
                    if df_agg.shape[0] > 1:
                        patch_tmp = np.zeros(
                            (
                                df_agg.shape[0],
                                patch.shape[1],
                                patch.shape[2],
                                patch.shape[3],
                            )
                        )
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
            img_montage[..., channel] = scale_image(
                img_montage[..., channel], range=(0, 255), percentile=1
            )
        img_montage = img_montage.astype(np.uint8)
        # save montage
        io.imsave(Path(dir_output, f'cluster_{cluster}_montage.png'), img_montage)
    montages = np.asarray(montages)
    montage_all = montage(montages, grid_shape=(1, len(clusters)), channel_axis=-1)
    for channel in range(montage_all.shape[-1]):
        montage_all[..., channel] = scale_image(
            montage_all[..., channel], range=(0, 255), percentile=1
        )
    io.imsave(Path(dir_output, 'montage_all.png'), montage_all.astype(np.uint8))


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

    ax[2].plot(
        model.history.history['reconstruction_loss'], label='reconstruction_loss'
    )
    ax[2].plot(
        model.history.history['val_reconstruction_loss'],
        label='val_reconstruction_loss',
    )
    ax[2].set_title('Reconstruction Loss')
    ax[2].legend()
    if show:
        plt.show()
    if return_fig:
        return fig
