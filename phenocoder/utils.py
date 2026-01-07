import io
import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from skimage.util import montage
from tqdm import tqdm


def plot_latent_space(
    model, generator, oh_enc, sample_frac=1, show=True, return_fig=False
):
    reducer = umap.UMAP()
    n_samples = int(sample_frac * len(generator))
    if n_samples == 0:
        n_samples = 1
    idx = np.random.choice(range(len(generator)), n_samples, replace=False)
    if generator.conditions is not None:
        data, conditions = zip(*[generator[i] for i in idx])
        data = np.concatenate(data, axis=0)
        conditions = np.concatenate(conditions, axis=0)
        z_mean, z_log_var, z = model.encoder.predict((data, conditions))
    else:
        generator.return_conditions = True
        data, conditions = zip(*[generator[i] for i in idx])
        data = np.concatenate(data, axis=0)
        conditions = np.concatenate(conditions, axis=0)
        z_mean, z_log_var, z = model.encoder.predict(data)

    df_labels = pd.DataFrame(
        oh_enc.inverse_transform(conditions), columns=oh_enc.feature_names_in_.tolist()
    )
    df_labels['z'] = pd.factorize(df_labels['z'])[0]
    z_umap = reducer.fit_transform(z)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    for i, dataset in enumerate(df_labels['dataset'].unique()):
        ax[0].scatter(
            z_umap[df_labels['dataset'] == dataset, 0],
            z_umap[df_labels['dataset'] == dataset, 1],
            label=dataset,
            s=0.5,
        )
    ax[0].legend()
    ax[0].set_title('dataset')
    scatter_z = ax[1].scatter(z_umap[:, 0], z_umap[:, 1], c=df_labels['z'], s=0.5)
    fig.colorbar(scatter_z, ax=ax[1])
    ax[1].set_title('z-stack position')
    plt.tight_layout()

    if show:
        plt.show()
    if return_fig:
        return fig


def plot_reconstructions(
    model, generator, n_preview=200, batch_size=64, show=True, return_fig=False
):
    if generator.conditions is not None:
        data, conditions = zip(
            *[generator[i] for i in range((n_preview // batch_size) + 1)]
        )
        data = np.concatenate(data, axis=0)
        conditions = np.concatenate(conditions, axis=0)
        z_mean, z_log_var, z = model.encoder.predict(
            (data, conditions), batch_size=batch_size
        )
        pred = model.decoder.predict([z, conditions], batch_size=batch_size)
    else:
        data = np.concatenate(
            [generator[i] for i in range((n_preview // batch_size) + 1)], axis=0
        )
        z_mean, z_log_var, z = model.encoder.predict(data, batch_size=batch_size)
        pred = model.decoder.predict(z, batch_size=batch_size)
    # sample n_preview images
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    idx = np.random.choice(
        range(data.shape[0]), n_preview, replace=n_preview > data.shape[0]
    )
    for i, ax in enumerate(axs.reshape(-1)):
        imgs_plot = np.concatenate([data[idx, :, :, 0], pred[idx, :, :, 0]], axis=2)
        # scale each patch to 0-1
        imgs_plot = np.asarray(
            [np.interp(img, (0, np.percentile(img, 99)), (0, 1)) for img in imgs_plot]
        )
        ax.imshow(montage(imgs_plot))
        ax.set_title(f'Channel {i}')
    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def write_training_plots_to_tensorboard(
    model,
    data_generator_train,
    model_oh_enc,
    dir_tensorboard,
    n_preview=300,
    plot_frac=0.001,
):
    """
    Write training visualization plots to TensorBoard.

    Generates and writes reconstruction and latent space visualization plots
    to TensorBoard for monitoring model training progress.

    Args:
        model: The trained model (CVAE or CondCVAE).
        data_generator_train: Training data generator.
        model_oh_enc: One-hot encoder for conditional models.
        dir_tensorboard (Path or str): Directory for TensorBoard logs.
        n_preview (int, optional): Number of samples for reconstruction plots. Defaults to 300.
        plot_frac (float, optional): Fraction of data for latent space visualization. Defaults to 0.001.

    Returns:
        None
    """
    file_writer = tf.summary.create_file_writer(str(dir_tensorboard))

    figure_reconstructions = plot_reconstructions(
        model,
        data_generator_train,
        n_preview=n_preview,
        return_fig=True,
        show=False,
    )
    with file_writer.as_default():
        tf.summary.image(
            'input vs reconstruction',
            plot_to_image(figure_reconstructions),
            step=0,
        )

    figure_latent_space = plot_latent_space(
        model,
        data_generator_train,
        model_oh_enc,
        sample_frac=plot_frac,
        return_fig=True,
        show=False,
    )
    with file_writer.as_default():
        tf.summary.image('latent space', plot_to_image(figure_latent_space), step=0)


def scale_image(
    image: np.ndarray, percentile: int = 1, range: tuple[int] = (0, 65535)
) -> np.ndarray:
    """
    Scale image
    :param image:
    :param percentile:
    :param range:
    :return:
    """
    image = np.interp(
        image,
        (np.percentile(image, percentile), np.percentile(image, 100 - percentile)),
        range,
    )
    return image


def load_features(
    well: str, dir_plate: str, cycle: str, input_type: str
) -> pd.DataFrame:
    """
    Load features for a given well
    :param well:
    :param dir_plate:
    :param cycle:
    :param input_type:
    :return:
    """
    # TODO: Refactor to accept sdata and load from sdata.tables instead of hard-coded directory structure
    # TODO: This function is dataset-specific with hard-coded paths (laminator, nuclei subdirs)
    dir_features = Path(dir_plate, cycle, 'features')
    dir_laminator = Path(dir_features, 'laminator', input_type, 'neighbors')
    dir_nuclei = Path(dir_features, 'nuclei', input_type)
    files = [f for f in os.listdir(dir_laminator) if f.startswith(f'{well}_')]
    if len(files) == 0:
        return pd.DataFrame()
    df_laminator = [
        pd.read_csv(Path(dir_laminator, f)).assign(
            z=f.replace(f'{well}_', '').replace('.csv', '')
        )
        for f in files
    ]
    if len(df_laminator) == 0:
        return pd.DataFrame()
    df_laminator = pd.concat(df_laminator)
    # drop columns starting with Unnamed
    df_laminator = df_laminator.loc[:, ~df_laminator.columns.str.contains('^Unnamed')]
    files = [f for f in os.listdir(dir_nuclei) if f.startswith(f'{well}_')]
    df_nuclei = [
        pd.read_csv(Path(dir_nuclei, f)).assign(
            z=f.replace(f'{well}_', '').replace('.csv', '')
        )
        for f in files
    ]
    df_nuclei = [df for df in df_nuclei if df.shape[0] > 0]
    if len(df_nuclei) == 0:
        return pd.DataFrame()
    df_nuclei = pd.concat(df_nuclei)
    df = pd.merge(df_laminator, df_nuclei, on=['label', 'z'], how='inner').set_index(
        'label'
    )
    df['z'] = df['z'].astype(int)
    df['z'] = df['z'] - 1
    # remove y, x, sample, well, z_stack columns
    columns_remove = ['y', 'x', 'sample', 'well', 'z_stack']
    df = df.drop(columns=columns_remove)
    # add neighbors suffix to all columns that start with ch_01
    columns = df.columns[df.columns.str.contains('^ch_0')]
    for column in columns:
        df.rename(columns={column: f'{column}_neighbors'}, inplace=True)
    # rename intensity_mean-0 to ch_01_intensity_mean
    columns = df.columns[df.columns.str.contains('intensity_mean')]
    # log1p columns
    df[columns] = np.log1p(df[columns])
    for column in columns:
        new_column = int(column.replace('intensity_mean-', '')) + 1
        df.rename(columns={column: f'ch_0{new_column}_nuclei'}, inplace=True)
    return df


def get_centroids(
    df: pd.DataFrame, z_step: int, pixel_size: float, filter_area: int = None
) -> pd.DataFrame:
    """
    Get centroids from label image
    :param df:
    :param z_step:
    :param pixel_size:
    :param filter_area:
    :return:
    """
    if df.empty:
        return df
    df['z_init'] = df['z']
    df['z'] = df['z'] / pixel_size * z_step
    df = df.groupby('label').mean()
    if filter_area is not None:
        df = df[df['area'] > filter_area]
    return df


def average_matched_nuclei(  # TODO: remove suffix prefix stuff
    adata: ad.AnnData, features: list, naming: str = 'suffix'
) -> ad.AnnData:
    """
    Average matched nuclei
    :param adata:
    :param features:
    :param naming:
    :return:
    """
    # TODO: Refactor to work with sdata.tables instead of manipulating adata.obs directly
    if naming == 'suffix':
        for feature in features:
            adata.obs[f'{feature}'] = adata.obs[
                [f'{feature}_source', f'{feature}_target']
            ].mean(axis=1)
    if naming == 'prefix':
        for feature in features:
            adata.obs[f'{feature}'] = adata.obs[
                [f'source_{feature}', f'target_{feature}']
            ].mean(axis=1)
    return adata


def load_plate(
    plate: str,
    input_type: str,
    dir_screen: str,
    registered: bool = True,
    plate_id: str = None,
    z_step: int = None,
) -> pd.DataFrame:
    """
    Load plate
    :param plate:
    :param input_type:
    :param dir_screen:
    :param registered:
    :param plate_id:
    :param z_step:
    :return:
    """
    # TODO: Refactor to accept sdata and load from sdata.tables instead of file system
    # TODO: This is completely file-based with hard-coded directory structures
    if registered:
        dir_registration = Path(dir_screen, plate, 'features_registration', input_type)
        files = os.listdir(dir_registration)
        df = [
            pd.read_csv(Path(dir_registration, file)).assign(
                well=file.replace('_registration.csv', '')
            )
            for file in tqdm(files, desc=f'Loading {plate}')
        ]

    else:
        if plate_id is None or z_step is None:
            raise ValueError(
                'plate_id and z_step must be provided for unregistered data!'
            )
        else:
            file = Path(
                dir_screen,
                plate,
                plate_id,
                'features',
                'nuclei',
                f'df_summary_{input_type}.csv',
            )
            wells = pd.read_csv(file)['well'].unique()
            df = [
                get_centroids(
                    load_features(
                        well,
                        dir_plate=Path(dir_screen, plate),
                        cycle=plate_id,
                        input_type=input_type,
                    ),
                    z_step=z_step,
                    pixel_size=0.322,
                ).assign(well=well)
                for well in tqdm(wells, desc=f'Loading {plate_id}')
            ]
    df = [df for df in df if not df.empty]
    df = pd.concat(df)
    df = df.assign(plate=plate)
    # index to unique string labels
    df.reset_index(drop=False, inplace=True)
    df.index = (
        df['label'].astype(str) + '_' + df['well'] + '_' + df['plate'].astype(str)
    )
    return df
