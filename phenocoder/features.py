from phenocoder.utils import suffix_to_prefix, load_plate, average_matched_nuclei
from phenocoder.phenocode import encode_nuclei_patches, merge_adata
from phenocoder.qc import qc_empty_wells
from pathlib import Path
import muon as mu
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import bbknn


def scale(adata: ad.AnnData) -> ad.AnnData:
    """
    Scale adata
    :param adata:
    :return:
    """
    adata = sc.pp.scale(adata, copy=True)
    return adata


def scale_per_plate(adata: ad.AnnData) -> ad.AnnData:
    """
    Scale per plate
    :param adata:
    :return:
    """
    plates = adata.obs['plate'].unique().tolist()
    adata = sc.concat(
        adatas=[scale(adata[adata.obs['plate'] == plate]) for plate in plates]
    )
    return adata


def add_morphology(adata: ad.AnnData) -> ad.AnnData:
    """
    Add morphology features
    :param adata:
    :return:
    """
    # extend with morphology features
    adata_morph = ad.AnnData(
        np.concatenate(
            [
                adata.X,
                adata.obs[
                    ['eccentricity', 'major_axis_length', 'minor_axis_length', 'area']
                ].values,
            ],
            axis=1,
        )
    )
    adata_morph.obs = adata.obs.drop(
        columns=['eccentricity', 'major_axis_length', 'minor_axis_length', 'area']
    )
    vars_nuc = adata.var_names.tolist()
    vars_nuc.extend(['eccentricity', 'major_axis_length', 'minor_axis_length', 'area'])
    adata_morph.var_names = vars_nuc
    return adata_morph


def process_features(
    df: pd.DataFrame,
    markers: dict,
    feature_type: str = None,
    scale_grouped: bool = False,
    morphology: bool = True,
    ridge: bool = True,
    registered: bool = True,
) -> ad.AnnData:
    """
    Process nuclei features
    :param df:
    :param markers:
    :param feature_type:
    :param scale_grouped:
    :param morphology:
    :param ridge:
    :param registered:
    :return:
    """
    if type is None:
        raise ValueError('feature_type must be either "nuclei" or "neighbors"')
    adata = ad.AnnData(df.filter(regex=f'_{feature_type}'))
    # harmonize to suffixes!
    if registered:
        adata.var_names = [markers[suffix_to_prefix(col)] for col in adata.var_names]
    else:
        adata.var_names = [markers[col] for col in adata.var_names]
    adata.obs = df.drop(columns=df.filter(regex=f'_{feature_type}').columns)
    if registered:
        print('Preprocessing registered data...')
        features = [
            'centroid-1',
            'centroid-0',
            'z',
            'eccentricity',
            'major_axis_length',
            'minor_axis_length',
            'area',
        ]
        adata = average_matched_nuclei(adata, features)

    if morphology:
        print('Adding morphology features...')
        adata = add_morphology(adata)

    # add X to layer raw
    adata.layers['X_raw'] = adata.X.copy()

    if scale_grouped:
        print('Scaling per plate...')
        adata = scale_per_plate(adata)
    else:
        print('Scaling...')
        adata = sc.pp.scale(adata, copy=True)

    if ridge:
        print('Running ridge regression...')
        adata = bbknn.ridge_regression(adata, batch_key=['plate_id', 'z'], copy=True)
        adata.X = adata.X.astype('float64')
        adata.layers['X_explained'] = adata.layers['X_explained'].astype('float64')

    return adata


def run_feature_processing(
    plates: list,
    markers: dict,
    dir_screen: str,
    input_type: str,
    registered: bool,
    dir_models: str,
    model_dict: dict,
    qc_score_threshold: int = 3,
    qc_distance_threshold: int = 200,
    n_wells_per_plate: int = None,
    cycle: str = None,
    channels: list = ['01', '02', '03', '04'],
):
    """
    Run feature processing
    :param plates:
    :param markers:
    :param dir_screen:
    :param input_type:
    :param registered:
    :param dir_models:
    :param model_dict:
    :param qc_score_threshold:
    :param qc_distance_threshold:
    :param n_wells_per_plate:
    :param cycle:
    """
    adata_pheno = []
    for plate in plates:
        if registered:
            print(f'Running QC for nuclei registration for plate {plate}...')
            df = load_plate(
                plate,
                input_type=input_type,
                dir_screen=dir_screen,
                registered=registered,
            )

            df = qc_empty_wells(
                df,
                dir_screen=dir_screen,
                threshold_score=qc_score_threshold,
                threshold_distance=qc_distance_threshold,
            )
            wells = df[df['plate'] == plate]['well'].unique().tolist()
        else:
            file = Path(
                dir_screen,
                plate,
                f'{plate}-{cycle}',
                'features',
                'nuclei',
                f'df_summary_{input_type}.csv',
            )
            wells = pd.read_csv(file)['well'].unique().tolist()

            qc_path = Path(
                dir_screen, plate, f'{plate}-{cycle}', f'{plate}-{cycle}_qc.csv'
            )
            if qc_path.is_file():
                qc_df = pd.read_csv(qc_path)
                bad_wells = qc_df[qc_df['decision'] == 'bad']['well_id'].tolist()
                wells = [well for well in wells if well not in bad_wells]

        print(f'Encoding nuclei patches for plate {plate}...')
        if n_wells_per_plate is not None:
            wells = wells[:n_wells_per_plate]
        if registered:
            adata_source = encode_nuclei_patches(
                well_ids=wells,
                plate=plate,
                dir_screen=dir_screen,
                cycle=model_dict['source']['cycle'],
                use_registered=True,
                label_type='source',
                dir_model=Path(dir_models, model_dict['source']['file']),
                channels=channels,
            )

            adata_target = encode_nuclei_patches(
                well_ids=wells,
                plate=plate,
                dir_screen=dir_screen,
                cycle=model_dict['target']['cycle'],
                use_registered=True,
                label_type='target',
                dir_model=Path(dir_models, model_dict['target']['file']),
                channels=channels,
            )

            adata = merge_adata(adata_source, adata_target)
            features_avg = [
                'centroid-1',
                'centroid-0',
                'z',
                'eccentricity',
                'major_axis_length',
                'minor_axis_length',
                'area',
            ]
            adata = average_matched_nuclei(adata, features_avg)

        else:
            if model_dict['source']['cycle'] == cycle:
                dir_model_cycle = model_dict['source']['file']
            elif model_dict['target']['cycle'] == cycle:
                dir_model_cycle = model_dict['target']['file']
            else:
                raise ValueError(f'Cycle {cycle} not found in model_dict')
            adata = encode_nuclei_patches(
                well_ids=wells,
                plate=plate,
                dir_screen=dir_screen,
                cycle=cycle,
                use_registered=False,
                dir_model=Path(dir_models, dir_model_cycle),
                channels=channels,
            )
        adata_pheno.append(adata)
    adata_pheno = sc.concat(adata_pheno)
    adata_pheno.obs.set_index('label', inplace=True)

    print('Processing nuclei features...')
    adata_nuc = process_features(
        adata_pheno.obs,
        markers,
        feature_type='nuclei',
        morphology=True,
        scale_grouped=False,
        ridge=True,
        registered=registered,
    )

    print('Processing neighborhood features...')
    adata_msg = process_features(
        adata_pheno.obs,
        markers,
        feature_type='neighbors',
        morphology=True,
        scale_grouped=False,
        ridge=True,
        registered=registered,
    )

    features_keep = ['well_id', 'plate_id', 'z', 'centroid-0', 'centroid-1']
    adata_pheno.obs = adata_pheno.obs[features_keep]
    adata_nuc.obs = adata_nuc.obs[features_keep]
    adata_msg.obs = adata_msg.obs[features_keep]
    mdata = mu.MuData(
        {'nuclei': adata_nuc, 'nuclei_msg': adata_msg, 'phenocoder': adata_pheno}
    )

    if 'phenocoder' in mdata.mod_names and 'phenocoder_msg' not in mdata.mod_names:
        if (
            np.sum(
                mdata['phenocoder'].obs.columns.isin(['z', 'centroid-0', 'centroid-1'])
            )
            != 3
        ):
            raise ValueError(
                'phenocoder modality must have z, centroid-0, centroid-1 columns'
            )
        mdata.mod['phenocoder_msg'] = mdata['phenocoder'].copy()
        mdata['phenocoder_msg'].X = (
            mdata['phenocoder_msg'].layers['message_passing'].copy()
        )
        mdata['phenocoder_msg'].layers = None
        mdata['phenocoder'].layers = None
        mdata.update()

    return mdata
