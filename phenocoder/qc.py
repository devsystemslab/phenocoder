import numpy as np
import anndata as ad
import pandas as pd
from skspatial.objects import Line, Points
from whole_mount_tumoroid.phenocoder.utils import average_matched_nuclei
import os
from pathlib import Path
from tqdm import tqdm

def get_count_cycle(well: str, plate: str, cycle: str,
                    dir_screen):
    """
    Get count for cycle
    :param well:
    :param plate:
    :param cycle:
    :param dir_screen:
    :return:
    """
    dir_features = Path(dir_screen, plate, f'{plate}-{cycle}', 'features', 'nuclei', 'TIF_OVR_BG')
    files = [file for file in os.listdir(dir_features) if file.startswith(f'{well}_')]
    if not files:
        return 0
    df = [pd.read_csv(Path(dir_features, file)) for file in files]
    # filter out empty dataframes and concatenate
    df = pd.concat([df for df in df if not df.empty])
    count = df['label'].nunique()
    return count


def fit_line_3d(adata: ad.AnnData, plate: str):
    """
    Fit line to 3d points
    :param adata:
    :param plate:
    :return:
    """
    adata = adata[adata.obs['plate'] == plate]
    results = []
    wells = adata.obs['well'].unique()
    for well in tqdm(wells, desc=f'Running 3d line fits for {plate}'):
        adata_well = adata[adata.obs['well'] == well]
        if adata_well.shape[0] < 2:
            results.append({'well': well, 'mean_distance': 0})
        else:
            points = Points(adata_well.obs[['z', 'centroid-0', 'centroid-1']].values)
            line_fit = Line.best_fit(points)
            mean_distance = line_fit.distance_points(points).mean()
            results.append({'well': well, 'mean_distance': mean_distance})
    # results to dataframe
    results = pd.DataFrame(results).assign(plate=plate)
    return results


def detect_2d_plane(adata, plate):
    """
    Detect 2d plane
    :param adata:
    :param plate:
    :return:
    """
    adata = adata[adata.obs['plate'] == plate]
    results = []
    wells = adata.obs['well'].unique()
    for well in tqdm(wells, desc=f'Running 2d plane detection for {plate}'):
        adata_well = adata[adata.obs['well'] == well]
        points = Points(adata_well.obs[['z', 'centroid-0', 'centroid-1']].values)
        for i in range(points.shape[-1]):
            if len(np.unique(points[:, i])) == 1:
                results.append({'well': well, 'plane_detected': True})
            else:
                results.append({'well': well, 'plane_detected': False})
    results = pd.DataFrame(results).assign(plate=plate)
    results = results.groupby(['well', 'plate']).agg({'plane_detected': 'any'}).reset_index()
    return results


def qc_empty_wells(df: pd.DataFrame, dir_screen: str,
                   threshold_distance: int = 100,
                   threshold_score: int = 3,
                   ):
    """
    QC detect empty and bad registered wells
    :param df:
    :param dir_screen:
    :param dir_screen:
    :param threshold_distance:
    :param threshold_score:
    :return:
    """

    adata = ad.AnnData(df.filter(regex='_nuclei'))
    # add obs all other columns that did not go into X
    adata.obs = df.drop(columns=df.filter(regex='_nuclei').columns)
    # average matched nuclei
    adata = average_matched_nuclei(adata,
                                   ['z', 'eccentricity', 'major_axis_length', 'minor_axis_length', 'area', 'centroid-0',
                                    'centroid-1'], naming='prefix')
    # set index to str
    adata.obs.index = adata.obs.index.astype('str')
    df_qc = adata.obs.groupby(['well', 'plate']).size().reset_index(name='count')
    plates = df_qc['plate'].unique().tolist()
    # remove wells with count == 0
    df_qc = df_qc[df_qc['count'] > 0]
    # linear regression
    df_linear_fit = pd.concat([fit_line_3d(adata, plate) for plate in plates])
    df_detect_plane = pd.concat([detect_2d_plane(adata, plate) for plate in plates])
    # merge with df_count
    df_qc = df_qc.merge(df_linear_fit, on=['well', 'plate'])
    df_qc = df_qc.merge(df_detect_plane, on=['well', 'plate'])
    tqdm.pandas(desc='Counting cycle 1')
    df_qc['count_cycle_1'] = df_qc.progress_apply(lambda x: get_count_cycle(x['well'], x['plate'], '01', dir_screen=dir_screen),
                                         axis=1)
    tqdm.pandas(desc='Counting cycle 3')
    df_qc['count_cycle_3'] = df_qc.progress_apply(lambda x: get_count_cycle(x['well'], x['plate'], '03', dir_screen=dir_screen),
                                         axis=1)
    df_qc['score'] = (df_qc['count_cycle_3'] + df_qc['count_cycle_1']) / (2 * df_qc['count'])
    df = df.reset_index().merge(df_qc, on=['well', 'plate'], how='left').drop(columns=['index'])
    df = df.reset_index(drop=True).rename(columns={'level_0': 'index'}).set_index('index')
    # filter out planes
    df = df[~df['plane_detected']]
    if threshold_score is not None:
        df = df[df['score'] < threshold_score]
    if threshold_distance is not None:
        df = df[df['mean_distance'] > threshold_distance]
    return df
