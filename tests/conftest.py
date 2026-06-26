import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from skimage import io

import phenocoder as phc


def example_3d():
    dir_tables = 'tests/data/3d/tables'
    table_files = sorted(os.listdir(dir_tables))
    df = pd.concat([pd.read_csv(Path(dir_tables, file)) for file in table_files])
    z_step = 10
    pixel_size = 0.322
    df['centroid-0'] = df['centroid-0'] / 4
    df['centroid-1'] = df['centroid-1'] / 4
    df['z_init'] = df['z_stack'] - 1
    df['z_stack'] = df['z_stack'] / pixel_size * z_step
    df = df.groupby(['label', 'well']).mean()
    df = df.reset_index()

    features_obs = [
        'area',
        'eccentricity',
        'major_axis_length',
        'minor_axis_length',
        'centroid-0',
        'centroid-1',
        'z_stack',
        'z_init',
        'well',
    ]

    features_X = [col for col in df.columns if col not in features_obs]
    adata = ad.AnnData(X=df[features_X], obs=df[features_obs])

    # Add spatial coordinates to adata.obsm
    adata.obsm['spatial'] = adata.obs[['centroid-0', 'centroid-1', 'z_stack']].values
    adata.obsm['spatial_index'] = adata.obs[
        ['centroid-0', 'centroid-1', 'z_init']
    ].values.astype(int)
    spatial_coords_2d = adata.obs[['centroid-0', 'centroid-1']].values
    adata.obsm['spatial_2d'] = spatial_coords_2d
    adata.obs['instance_id'] = adata.obs.index.astype(int)
    adata.obs['region'] = 'nuclei'

    dir_images = 'tests/data/3d/imgs'
    img_files = sorted(os.listdir(dir_images))
    channels = [f'C0{i + 1}' for i in range(4)]
    wells = adata.obs['well'].unique()

    images_dict = {}
    labels_dict = {}

    # build per-well image and label entries
    for well in wells:
        # select image files belonging to this well
        files_well = [f for f in img_files if f'_{well}_' in f]

        imgs = []
        for channel in channels:
            files_channel = [
                file for file in files_well if file.endswith(f'{channel}.png')
            ]
            imgs.append(
                np.asarray(
                    [io.imread(Path(dir_images, file)) for file in files_channel]
                )
            )
        imgs = np.asarray(imgs)

        # select label files for this well
        dir_labels = 'tests/data/3d/labels'
        img_label_files = sorted(os.listdir(dir_labels))
        img_label_files_well = [f for f in img_label_files if f'_{well}_' in f]
        imgs_label = np.asarray(
            [io.imread(Path(dir_labels, file)) for file in img_label_files_well]
        )

        images_dict[f'IF_{well}'] = sd.models.Image3DModel.parse(
            imgs, c_coords=channels
        )
        images_dict[f'IF_MIP_{well}'] = sd.models.Image2DModel.parse(
            imgs.max(axis=1), c_coords=channels
        )
        labels_dict[f'nuclei_{well}'] = sd.models.Labels3DModel.parse(imgs_label)

    sdata = sd.SpatialData(
        images=images_dict,
        labels=labels_dict,
        tables={
            'nuclei_features': sd.models.TableModel.parse(
                adata,
                region='nuclei',
                region_key='region',
                instance_key='instance_id',
                overwrite_metadata=True,
            )
        },
    )

    # setup phenocoder
    pheno = phc.Phenocoder(
        table_key='nuclei_features',
        sample_key='well',
        image_key='IF',
        labels_key='nuclei',
    )
    pheno.add_sdata(sdata)

    return pheno


@pytest.fixture
def phenocoder_3d():
    pheno = example_3d()
    return pheno
