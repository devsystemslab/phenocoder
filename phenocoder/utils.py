from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from skimage.util import montage

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from sklearn.preprocessing import OneHotEncoder

    from phenocoder.generator import SequenceGenerator
    from phenocoder.model import CVAE, CondCVAE


def plot_latent_space(
    model: "CVAE | CondCVAE",
    generator: "SequenceGenerator",
    oh_enc: "OneHotEncoder",
    sample_frac: float = 1,
    show: bool = True,
    return_fig: bool = False,
) -> "Figure | None":
    """
    Plot UMAP visualization of the latent space colored by dataset and z-position.

    Generates a 2D UMAP projection of the latent space representations and creates
    two scatter plots: one colored by dataset and one colored by z-stack position.

    Args:
        model: Trained CVAE or CondCVAE model with encoder.
        generator: Data generator (SequenceGenerator) providing image patches.
        oh_enc: One-hot encoder used for encoding conditions.
        sample_frac (float, optional): Fraction of generator batches to sample for plotting.
            Defaults to 1 (use all data).
        show (bool, optional): Whether to display the plot. Defaults to True.
        return_fig (bool, optional): Whether to return the figure object. Defaults to False.

    Returns:
        matplotlib.figure.Figure or None: Figure object if return_fig=True, otherwise None.

    """
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
    z_umap = reducer.fit_transform(z)

    # one panel per condition the model was trained on (not hardcoded to a
    # 'dataset'/'z' pair). A 'z' column is shown as a continuous z-stack gradient;
    # any other condition is shown as discrete groups with a legend.
    cols = df_labels.columns.tolist()
    fig, axs = plt.subplots(ncols=len(cols), figsize=(6 * len(cols), 6), squeeze=False)
    for ax, col in zip(axs.reshape(-1), cols):
        if col == 'z':
            codes = pd.factorize(df_labels[col])[0]
            scatter_z = ax.scatter(z_umap[:, 0], z_umap[:, 1], c=codes, s=0.5)
            fig.colorbar(scatter_z, ax=ax)
            ax.set_title('z-stack position')
        else:
            for group in df_labels[col].unique():
                mask = (df_labels[col] == group).values
                ax.scatter(z_umap[mask, 0], z_umap[mask, 1], label=group, s=0.5)
            ax.legend()
            ax.set_title(col)
    plt.tight_layout()

    if show:
        plt.show()
    if return_fig:
        return fig


def plot_reconstructions(
    model: "CVAE | CondCVAE",
    generator: "SequenceGenerator",
    n_preview: int = 200,
    batch_size: int = 64,
    show: bool = True,
    return_fig: bool = False,
) -> "Figure | None":
    """
    Plot side-by-side comparison of input images and their VAE reconstructions.

    Creates a montage visualization showing original input patches alongside their
    reconstructions from the VAE model. Displays the first channel for all selected patches.

    Args:
        model: Trained CVAE or CondCVAE model with encoder and decoder.
        generator: Data generator (SequenceGenerator) providing image patches.
        n_preview (int, optional): Number of image patches to visualize. Defaults to 200.
        batch_size (int, optional): Batch size for model predictions. Defaults to 64.
        show (bool, optional): Whether to display the plot. Defaults to True.
        return_fig (bool, optional): Whether to return the figure object. Defaults to False.

    Returns:
        matplotlib.figure.Figure or None: Figure object if return_fig=True, otherwise None.
    """
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
    n_channels = data.shape[-1]
    fig, axs = plt.subplots(n_channels, 1, figsize=(10, 5 * n_channels), squeeze=False)
    idx = np.random.choice(
        range(data.shape[0]), n_preview, replace=n_preview > data.shape[0]
    )
    for i, ax in enumerate(axs.reshape(-1)):
        imgs_plot = np.concatenate([data[idx, :, :, i], pred[idx, :, :, i]], axis=2)
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


def plot_to_image(figure: "Figure") -> tf.Tensor:
    """
    Convert a matplotlib figure to a TensorFlow image tensor.

    Converts the matplotlib plot to a PNG image in memory and returns it as a
    TensorFlow tensor suitable for TensorBoard logging. The supplied figure is
    closed and inaccessible after this call.

    Args:
        figure (matplotlib.figure.Figure): Matplotlib figure to convert.

    Returns:
        tf.Tensor: TensorFlow image tensor with shape (1, height, width, 4) in RGBA format.
    """
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
    model: "CVAE | CondCVAE",
    data_generator_train: "SequenceGenerator",
    model_oh_enc: "OneHotEncoder",
    dir_tensorboard: Path | str,
    n_preview: int = 300,
    plot_frac: float = 0.001,
) -> None:
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


def _coerce_stringdtype_uns(adata: ad.AnnData) -> None:
    """Coerce NumPy ``StringDType`` arrays in ``adata.uns`` to object dtype, in place.

    scanpy stores categorical color palettes (e.g. ``<key>_colors``) in ``uns`` and
    anndata reads them back as NumPy 2.0 ``StringDType`` arrays (``dtype.kind == 'T'``).
    Those arrays segfault under ``copy.deepcopy``, which ``AnnData.copy()`` uses when
    subsetting per sample. Coercing to object dtype is harmless for plotting and
    serialization and makes the table safe to copy.
    """
    for key, val in list(adata.uns.items()):
        if isinstance(val, np.ndarray) and val.dtype.kind == 'T':
            adata.uns[key] = np.asarray(val, dtype=object)
