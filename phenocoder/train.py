import sys
sys.path.append('/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen')
import pandas as pd
import os
import keras
from whole_mount_tumoroid.phenocoder.model import CVAE, CondCVAE
from whole_mount_tumoroid.phenocoder.generator import setup_generators
import yaml
import tensorflow as tf
import joblib
import umap
import numpy as np
import matplotlib.pyplot as plt
import io
from skimage.util import montage


def plot_latent_space(model, generator, oh_enc, sample_frac=1, show=True, return_fig=False):
    reducer = umap.UMAP()
    n_samples = int(sample_frac * len(generator))
    idx = np.random.choice(range(len(generator)), n_samples, replace=False)
    if generator.return_conditions:
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

    df_labels = pd.DataFrame(oh_enc.inverse_transform(conditions), columns=oh_enc.feature_names_in_.tolist())
    df_labels['z'] = pd.factorize(df_labels['z'])[0]
    z_umap = reducer.fit_transform(z)
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    for i, dataset in enumerate(df_labels['dataset'].unique()):
        ax[0].scatter(z_umap[df_labels['dataset'] == dataset, 0],
                      z_umap[df_labels['dataset'] == dataset, 1],
                      label=dataset,
                      s=0.5)
    ax[0].legend()
    ax[0].set_title('dataset')
    scatter_z = ax[1].scatter(z_umap[:, 0],
                              z_umap[:, 1],
                              c=df_labels['z'],
                              s=0.5)
    fig.colorbar(scatter_z, ax=ax[1])
    ax[1].set_title('z-stack position')
    plt.tight_layout()

    if show:
        plt.show()
    if return_fig:
        return fig

def plot_reconstructions(model, generator, n_preview=200, batch_size=64, show=True, return_fig=False):
    if generator.return_conditions:
        data, conditions = zip(*[generator[i] for i in range((n_preview // batch_size)+1)])
        data = np.concatenate(data, axis=0)
        conditions = np.concatenate(conditions, axis=0)
        z_mean, z_log_var, z = model.encoder.predict((data, conditions), batch_size=batch_size)
        pred = model.decoder.predict([z, conditions], batch_size=batch_size)
    else:
        data = np.concatenate([generator[i] for i in range((n_preview // batch_size)+1)], axis=0)
        z_mean, z_log_var, z = model.encoder.predict(data, batch_size=batch_size)
        pred = model.decoder.predict(z, batch_size=batch_size)
    # sample n_preview images
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    idx = np.random.choice(range(data.shape[0]), n_preview, replace=n_preview > data.shape[0])
    for i, ax in enumerate(axs.reshape(-1)):
        imgs_plot = np.concatenate([data[idx, :, :, 0], pred[idx, :, :, 0]], axis=2)
        # scale each patch to 0-1
        imgs_plot = np.asarray([np.interp(img, (0, np.percentile(img, 99)), (0, 1)) for img in imgs_plot])
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


def train_model(dir_dataset,
                n_latent_dim,
                n_dense_dim,
                n_epochs,
                conditional,
                dropout=0.25,
                batch_size=64,
                n_workers=1,
                input_shape=(128,128,4),
                conv_layers=(8,16,32,64,128),
                beta=1,
                plot_frac=0.1):
    """
    Train CVAE model
    :param dir_dataset:
    :param n_latent_dim:
    :param n_dense_dim:
    :param n_epochs:
    :param conditional:
    :param dropout:
    :param batch_size:
    :param n_workers:
    :param input_shape:
    :param conv_layers:
    :param beta:
    :param plot_frac:
    :return:
    """
    # define model name
    model_name = f'latent_{n_latent_dim}_dense_{n_dense_dim}_dropout_{dropout}_beta_{beta}_{pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")}'
    if conditional:
        model_name = f'cond_{model_name}'
    # set up model directory
    dir_model = os.path.join(dir_dataset, 'models', model_name)
    os.makedirs(dir_model, exist_ok=True)

    dim = input_shape[:2]
    n_channels = input_shape[2]

    # set up data generators
    generator_train, generator_val, df_cond, oh_enc = setup_generators(dir_dataset,
                                                                       conditional,
                                                                       batch_size=batch_size,
                                                                       n_workers=n_workers,
                                                                       dim=dim,
                                                                       n_channels=n_channels,
                                                                       shuffle=True)

    # write all parameters to yaml
    param_dict = {'n_latent_dim': n_latent_dim,
                  'n_dense_dim': n_dense_dim,
                  'n_epochs': n_epochs,
                  'input_shape': list(input_shape),
                  'conv_layers': list(conv_layers),
                  'conditional': conditional,
                  'dropout': dropout,
                  'dir_dataset': dir_dataset,
                  'batch_size': batch_size,
                  'n_workers': n_workers,
                  'beta': beta,
                  'quantiles_low': generator_train.quantiles_low.tolist(),
                  'quantiles_high': generator_train.quantiles_high.tolist(),
                  'conditions_dim': generator_train.conditions.shape[-1]}
    # write to yaml
    with open(os.path.join(dir_model,'config.yaml'),'w') as file:
        yaml.dump(param_dict, file)
    joblib.dump(oh_enc, os.path.join(dir_model,'oh_encoder.joblib'))
    # write df conditions to file
    df_cond.to_csv(os.path.join(dir_model,'df_files.csv'), index=False)

    # set up model
    if conditional:
        cvae = CondCVAE(n_classes=generator_train.conditions.shape[-1],
                        input_shape=input_shape,
                        latent_dim=n_latent_dim,
                        dense_dim=n_dense_dim,
                        conv_layers=conv_layers,
                        dropout=dropout,
                        beta=beta)
    else:
        cvae = CVAE(input_shape=input_shape,
                    latent_dim=n_latent_dim,
                    dense_dim=n_dense_dim,
                    conv_layers=conv_layers,
                    dropout=dropout,
                    beta=beta)

    # set up tensorboard
    dir_tensorboard = os.path.join(dir_dataset, 'tensorboard_logs', model_name)
    if not os.path.exists(dir_tensorboard):
        os.makedirs(dir_tensorboard)
    print('Tensorflow devices')
    print(tf.config.list_physical_devices())
    # model summary
    print('Model summary:')
    for key, value in param_dict.items():
        print(f'{key}: {value}')
    cvae.encoder.summary()
    cvae.decoder.summary()
    # compile
    cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    # train
    cvae.fit(generator_train,
             validation_data=generator_val,
             epochs=n_epochs,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        keras.callbacks.TensorBoard(log_dir=dir_tensorboard),
                        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)])

    #build model before saving
    cvae.build(input_shape)

    # save weights
    cvae.save_weights(os.path.join(dir_model, 'model.weights.h5'))
    # add plots to tensorboard
    file_writer = tf.summary.create_file_writer(dir_tensorboard)
    # prepare the plot
    print('Preparing plots for tensorboard...')
    figure_reconstructions = plot_reconstructions(cvae, generator_train, n_preview=300, return_fig=True, show=False)
    with file_writer.as_default():
        tf.summary.image("input vs reconstruction", plot_to_image(figure_reconstructions), step=0)

    figure_latent_space = plot_latent_space(cvae, generator_train, oh_enc, sample_frac=plot_frac,
                                            return_fig=True, show=False)
    with file_writer.as_default():
        tf.summary.image("latent space", plot_to_image(figure_latent_space), step=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('dir_dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--n_latent_dim', type=int, default=128)
    parser.add_argument('--n_dense_dim', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--input_shape', type=int, nargs=3, default=[128, 128, 4])
    parser.add_argument('--conditional', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--plot_frac', type=float, default=0.1)
    args = parser.parse_args()

    input_shape = tuple(args.input_shape)

    train_model(dir_dataset=args.dir_dataset,
                n_latent_dim=args.n_latent_dim,
                n_dense_dim=args.n_dense_dim,
                n_epochs=args.n_epochs,
                input_shape=args.input_shape,
                conditional=args.conditional,
                dropout=args.dropout,
                batch_size=args.batch_size,
                n_workers=args.n_workers,
                beta=args.beta,
                plot_frac=args.plot_frac)
