from phenocoder._phenocode import encode_nuclei_patches


def test_predict():
    encode_nuclei_patches(
        well='A01',
        plate='HM003',
        dir_screen='/pstore/data/ihb-tumoroidscreen/data/processed/tumoroidscreen',
        cycle='01',
        model_config='/scratch/USERS/harmelc/models/model_config.yaml',
        model_weights='/scratch/USERS/harmelc/models/model_weights.h5',
        input_type='TIF_OVR_BG',
    )
