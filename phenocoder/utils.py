import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import umap
from skimage.util import montage


def plot_latent_space(
    model, generator, oh_enc, sample_frac=1, show=True, return_fig=False
):
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

    Note:
        TODO: generalize plotting function -> add argument for conditions to plot.
    """
    # TODO: generalize plotting function -> add argument for conditions to plot.
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
