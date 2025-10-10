import numpy as np
import pandas as pd
import os
from pathlib import Path
import anndata as ad
from tqdm import tqdm


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
