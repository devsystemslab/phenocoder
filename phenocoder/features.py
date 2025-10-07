from phenocoder.utils import average_matched_nuclei
from phenocoder.phenocode import encode_nuclei_patches
from pathlib import Path
import muon as mu
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc


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
    adata.var_names = [markers[col] for col in adata.var_names]
    adata.obs = df.drop(columns=df.filter(regex=f'_{feature_type}').columns)
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

    return adata


def run_feature_processing(
    plates: list,
    markers: dict,
    dir_screen: str,
    input_type: str,
    dir_models: str,
    model_dict: dict,
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
    :param dir_models:
    :param model_dict:
    :param n_wells_per_plate:
    :param cycle:
    :param channels:
    """
    adata_pheno = []
    for plate in plates:
        file = Path(
            dir_screen,
            plate,
            f'{plate}-{cycle}',
            'features',
            'nuclei',
            f'df_summary_{input_type}.csv',
        )
        wells = pd.read_csv(file)['well'].unique().tolist()

        qc_path = Path(dir_screen, plate, f'{plate}-{cycle}', f'{plate}-{cycle}_qc.csv')
        if qc_path.is_file():
            qc_df = pd.read_csv(qc_path)
            bad_wells = qc_df[qc_df['decision'] == 'bad']['well_id'].tolist()
            wells = [well for well in wells if well not in bad_wells]

        print(f'Encoding nuclei patches for plate {plate}...')
        if n_wells_per_plate is not None:
            wells = wells[:n_wells_per_plate]
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
    )

    print('Processing neighborhood features...')
    adata_msg = process_features(
        adata_pheno.obs,
        markers,
        feature_type='neighbors',
        morphology=True,
        scale_grouped=False,
        ridge=True,
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
