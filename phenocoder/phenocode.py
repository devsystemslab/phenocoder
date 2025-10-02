from phenocoder.model import CVAE, CondCVAE
from phenocoder.utils import load_features
from phenocoder.generator import NucleiPatchGenerator, GridPatchGenerator
import yaml
import pandas as pd
import anndata as ad
import joblib
import numpy as np
import squidpy as sq
import re
from scipy.sparse import csr_array


def load_config(model_config):
    """
    Load model config
    :param model_config:
    :return:
    """
    with open(model_config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_one_hot_encoder(file):
    """
    Load hot encoder
    :param file:
    :return:
    """
    return joblib.load(file)


def load_phenocoder(directory):
    """
    Load phenocoder model
    :param directory:
    :return:
    """
    config = load_config(f'{directory}/config.yaml')
    if config['conditional']:
        model = CondCVAE(
            input_shape=tuple(config['input_shape']),
            latent_dim=config['n_latent_dim'],
            dense_dim=config['n_dense_dim'],
            conv_layers=tuple(config['conv_layers']),
            n_classes=config['conditions_dim'],
        )
    else:
        model = CVAE(
            input_shape=tuple(config['input_shape']),
            latent_dim=config['n_latent_dim'],
            dense_dim=config['n_dense_dim'],
            conv_layers=tuple(config['conv_layers']),
        )
    model.compile()
    model.load_weights(f'{directory}/model.weights.h5')
    oh_enc = load_one_hot_encoder(f'{directory}/oh_encoder.joblib')

    return model, oh_enc, config


def encode_nuclei_patches(
    well_ids,
    plate,
    dir_screen,
    cycle,
    dir_model,
    input_type='TIF_OVR_BG',
    use_registered=False,
    label_type=None,
    df_labels=None,
    return_only_patches=False,
    filter_encodable_conditions=False,
    message_passing=True,
    radius=100,
    channels=['01', '02', '03', '04'],
):
    """
    Encode nuclei patches from a well in a plate
    :param well_ids:
    :param plate:
    :param dir_screen:
    :param cycle:
    :param dir_model:
    :param input_type:
    :param use_registered:
    :param label_type:
    :param df_labels:
    :param return_only_patches:
    :param filter_encodable_conditions:
    :param message_passing:
    :param radius:
    :return:
    """
    # setup directories
    dir_plate = f'{dir_screen}/{plate}'
    cycle = f'{plate}-{cycle}'
    # load model
    cvae, oh_enc, config = load_phenocoder(dir_model)
    # set up patch generator
    nuclei_patch_generator = NucleiPatchGenerator(
        f'{dir_screen}/{plate}/{cycle}/{input_type}'
    )
    results = []
    for well in well_ids:
        if use_registered:
            file = f'{dir_plate}/features_registration/{input_type}/{well}_registration.csv'
            df = pd.read_csv(file).assign(well_id=well, plate_id=cycle)
            if label_type is not None:
                # select all columns that start with label_type
                cols = df.filter(regex=f'^{label_type}').columns.tolist()
                cols.extend(['label', 'well_id', 'plate_id'])
                # select columns and also keep well_id and plate_id
                df = df[cols]
                # rename label_type label to label_cycle
                df = df.rename(columns={f'{label_type}_label': 'label_cycle'})
                # remove label type from all colnames
                df.columns = df.columns.str.replace(f'{label_type}_', '')
                # convert pixel to z-stack id
                df['z'] = df['z'] * 0.322 / 10
                # convert to int
                df['z'] = df['z'].astype(int) + 1
                # set label as index
                df = df.set_index('label')
            else:
                raise ValueError(
                    'Label type must be provided when using registered data'
                )
        else:
            df = load_features(
                well, dir_plate=dir_plate, cycle=cycle, input_type=input_type
            ).assign(well_id=well, plate_id=cycle)
            channel_pattern = r'ch_(' + '|'.join(channels) + r')_'
            general_pattern = r'ch_\d{2}_'
            # drop nuclei and neighbour columns of channels that are not used
            df = df[
                [
                    col
                    for col in df.columns
                    if not re.match(general_pattern, col)
                    or re.search(channel_pattern, col)
                ]
            ]
            if df.empty:
                continue
            df['z'] = df['z'] + 1
        if df_labels is not None:
            df_labels_well = df_labels[df_labels['well_id'] == well]
            # filter df for labels in df_labels_well
            df = df[df.index.isin(df_labels_well['label'].astype(int))]
        patches, df = nuclei_patch_generator.get_patches(
            well,
            df,
            scale=True,
            quantiles_low=config['quantiles_low'],
            quantiles_high=config['quantiles_high'],
            channels=channels,
        )
        df = df.reset_index()
        if df.empty:
            continue
        if return_only_patches:
            results.append([patches, df])
        else:
            if config['conditional']:
                df_cond = df[['plate_id', 'z']]
                # rename plate_id to dataset
                df_cond = df_cond.rename(columns={'plate_id': 'dataset'})
                # reset index
                if filter_encodable_conditions:
                    # filter out conditions which cannot be encoded
                    idx = df_cond.index[
                        (df_cond['z'].isin(oh_enc.categories_[1]))
                        & (df_cond['dataset'].isin(oh_enc.categories_[0]))
                    ]
                    conditions = oh_enc.transform(
                        df_cond.iloc[idx][oh_enc.feature_names_in_]
                    )
                    patches = patches[idx]
                    df = df.iloc[idx]
                else:
                    conditions = oh_enc.transform(df_cond[oh_enc.feature_names_in_])
                z_mean, z_log_var, z = cvae.encoder.predict([patches, conditions])
            else:
                z_mean, z_log_var, z = cvae.encoder.predict(patches, batch_size=64)
            df_z = pd.DataFrame(z, columns=[f'z_{i}' for i in range(z.shape[-1])])
            # reset z to pixel coordinates
            df['z'] = df['z'] / 0.322 * 10

            if not use_registered:
                # drop non numeric cols
                df = df.drop(columns=['well_id', 'plate_id'])
                # add df_z to df
                df = pd.concat([df, df_z], axis=1)
                df = df.groupby('label').mean()
                df = df.reset_index()
                df_z = df[[f'z_{i}' for i in range(z.shape[-1])]]
                df = df.drop(columns=[f'z_{i}' for i in range(z.shape[-1])])

            df = df.assign(well_id=well, plate_id=plate)

            df.index = (
                df['label'].astype(str) + '_' + df['well_id'] + '_' + df['plate_id']
            )

            adata = ad.AnnData(
                X=df_z.values,
                obs=df,
                var=pd.DataFrame(index=[f'z_{i + 1}' for i in range(df_z.shape[-1])]),
            )

            if message_passing:
                # calculate knn graph in physical space
                adata.obsm['spatial'] = adata.obs[['x', 'y', 'z']].values.copy()
                sq.gr.spatial_neighbors(
                    adata, radius=radius, coord_type='generic', spatial_key='spatial'
                )
                A = adata.obsp['spatial_connectivities'].copy()
                A = A + csr_array(np.diag(np.ones(A.shape[0])))
                # weight A with inverse degree matrix
                D = np.array(A.sum(axis=1)).flatten()
                D_inv = np.power(D, -1)
                D_inv[np.isinf(D_inv)] = 0
                D_inv = np.diag(D_inv)
                A = A.dot(D_inv)
                adata.layers['message_passing'] = np.dot(A, adata.X)
            results.append(adata)
    if return_only_patches:
        return results
    else:
        adata = ad.concat(results)
        adata.obs['label'] = adata.obs.index.copy()
    return adata


def merge_adata(adata_source, adata_target, make_labels_unique=False):
    """
    Merge adata_source and adata_target
    :param adata_source:
    :param adata_target:
    :param make_labels_unique:
    :return:
    """
    if make_labels_unique:
        adata_source.obs['label'] = (
            adata_source.obs['label'].astype(str) + '_' + adata_source.obs['well_id']
        )
        adata_target.obs['label'] = (
            adata_target.obs['label'].astype(str) + '_' + adata_target.obs['well_id']
        )
    labels_target = adata_target.obs['label'].values
    labels_source = adata_source.obs['label'].values
    labels = list(set(labels_target).intersection(set(labels_source)))
    # select labels in adata_target and source
    idx_target = adata_target.obs.index[adata_target.obs['label'].isin(labels)]
    idx_source = adata_source.obs.index[adata_source.obs['label'].isin(labels)]

    adata_target = adata_target[idx_target]
    adata_source = adata_source[idx_source]
    df = adata_target.obs.merge(
        adata_source.obs,
        on=['label', 'well_id', 'plate_id'],
        suffixes=('_target', '_source'),
    )
    adata = ad.AnnData(X=np.concatenate([adata_target.X, adata_source.X], axis=1))
    adata.var_names = [name + '_target' for name in adata_target.var_names] + [
        name + '_source' for name in adata_source.var_names
    ]
    adata.layers['message_passing'] = np.concatenate(
        [
            adata_target.layers['message_passing'],
            adata_source.layers['message_passing'],
        ],
        axis=1,
    )
    adata.obs = df

    return adata


def encode_grid_patches(
    well,
    plate,
    dir_screen,
    cycle,
    dir_model,
    input_type='TIF_OVR_BG',
    grid_resolution=1,
    filter_encodable_conditions=False,
    concatenate_latent_variables=False,
):
    """
    Encode grid patches
    :param well:
    :param plate:
    :param dir_screen:
    :param cycle:
    :param dir_model:
    :param input_type:
    :param grid_resolution:
    :param filter_encodable_conditions:
    :param concatenate_latent_variables:
    :return:
    """
    cycle = f'{plate}-{cycle}'
    cvae, oh_enc, config = load_phenocoder(dir_model)
    grid_patch_generator = GridPatchGenerator(
        f'{dir_screen}/{plate}/{cycle}/{input_type}',
        stride=int(tuple(config['input_shape'])[0] / grid_resolution),
    )
    patches, df = grid_patch_generator.get_patches(
        well, quantiles=config['quantiles_high']
    )
    df = df.assign(plate_id=cycle)
    df['z'] = df['z_stack_id'].astype(int)
    # get z, x, y idx coordinates
    if config['conditional']:
        df_cond = df[['plate_id', 'z']]
        # rename plate_id to dataset
        df_cond = df_cond.rename(columns={'plate_id': 'dataset'})
        # reset index
        if filter_encodable_conditions:
            # filter out conditions which cannot be encoded
            idx = df_cond.index[
                (df_cond['z'].isin(oh_enc.categories_[1]))
                & (df_cond['dataset'].isin(oh_enc.categories_[0]))
            ]
            conditions = oh_enc.transform(
                df_cond.iloc[idx][oh_enc.feature_names_in_]
            ).toarray()
            patches = patches[idx]
            df = df.iloc[idx]
        else:
            conditions = oh_enc.transform(df_cond[oh_enc.feature_names_in_]).toarray()
        z_mean, z_log_var, z = cvae.encoder.predict(
            [patches, conditions], batch_size=64
        )
    else:
        z_mean, z_log_var, z = cvae.encoder.predict(patches, batch_size=64)
    if concatenate_latent_variables:
        z = np.concatenate([z_mean, z_log_var, z], axis=-1)
    df_z = pd.DataFrame(z, columns=[f'z_{i}' for i in range(z.shape[-1])])
    df = pd.concat([df.reset_index(), df_z.reset_index()], axis=1)
    adata = ad.AnnData(X=df[[f'z_{i}' for i in range(z.shape[-1])]].values)
    adata.obs = df[['x', 'y', 'z']].assign(well_id=well, plate_id=plate)

    return adata
