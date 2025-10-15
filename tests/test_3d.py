import spatialdata as sd
import phenocoder as phc
import numpy as np
import anndata as ad
from skimage import io
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_centroids(
    df: pd.DataFrame, z_step: int = 10, pixel_size: float = 0.322
) -> pd.DataFrame:
    """
    Get centroids from label image
    :param df:
    :param z_step:
    :return:
    """
    if df.empty:
        return df
    well = sorted(df['well'].unique())[0]
    df = df.drop('well', axis=1)
    df['z_init'] = df['z_stack']
    df['z_stack'] = df['z_stack'] / pixel_size * z_step
    df = df.groupby('label').mean()
    df['well'] = well
    return df


dir_images = 'tests/data/3d/imgs'
img_files = sorted(os.listdir(dir_images))

channels = [f'C0{i + 1}' for i in range(4)]
imgs = []
for channel in channels:
    files_channel = [file for file in img_files if file.endswith(f'{channel}.png')]
    imgs.append(
        np.asarray([io.imread(Path(dir_images, file)) for file in files_channel])
    )
imgs = np.asarray(imgs)

dir_labels = 'tests/data/3d/labels'
img_label_files = sorted(os.listdir(dir_labels))
imgs_label = np.asarray([io.imread(Path(dir_labels, file)) for file in img_label_files])

dir_tables = 'tests/data/3d/tables'
table_files = sorted(os.listdir(dir_tables))
df_features = pd.concat([pd.read_csv(Path(dir_tables, file)) for file in table_files])
df_features = get_centroids(df_features)

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

features_X = [col for col in df_features.columns if col not in features_obs]
adata = ad.AnnData(X=df_features[features_X], obs=df_features[features_obs])
sdata = sd.SpatialData(
    images={
        'IF': sd.models.Image3DModel.parse(imgs, c_coords=channels),
        'IF_MIP': sd.models.Image2DModel.parse(imgs.max(axis=1), c_coords=channels),
    },
    labels={'nuclei': sd.models.Labels3DModel.parse(imgs_label)},
    tables={'nuclei_features': sd.models.TableModel.parse(adata)},
)
sdata.write(Path('tests/data/3d', 'sdata'), overwrite=True)

# setup phenocoder
pheno = phc.Phenocoder()
pheno.add_sdata(sdata)
