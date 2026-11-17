"""Encoding needs only the model directory, and copes with samples the model never saw."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import example_3d

DIR_PROJECT = Path('tests/data/tmp_encode')


@pytest.fixture(scope='module')
def config_path():
    """A conditional model trained one epoch on every well, as ``dataset_1``."""
    pheno = example_3d()
    pheno.dir_project = DIR_PROJECT
    try:
        pheno.generate_dataset(
            dataset='dataset_1', patch_size=(32, 32), spatial_key_index='spatial_index'
        )
        pheno.initialize_model(
            n_latent_dim=8,
            n_dense_dim=16,
            conditions=['dataset', 'z'],
            input_shape=(32, 32, 4),
        )
        pheno.train(n_epochs=1, plot=False)
        yield Path(pheno.model_dir, 'config.yaml')
    finally:
        shutil.rmtree(DIR_PROJECT, ignore_errors=True)


def _loaded(config_path, pheno=None):
    pheno = pheno or example_3d()
    pheno.model_config = str(config_path)
    pheno.load_model()
    return pheno


def test_model_directory_keeps_what_encode_needs(config_path):
    model_dir = config_path.parent
    assert Path(model_dir, 'stats.csv').exists()
    datasets = Path(model_dir, 'sample_datasets.csv').read_text().splitlines()
    assert datasets[0] == 'well,dataset'
    assert all(line.endswith(',dataset_1') for line in datasets[1:])


def test_encode_without_the_training_dataset(config_path, tmp_path):
    """The dataset directory is moved away: encode reads the model directory instead."""
    dataset_dir = Path(DIR_PROJECT, 'dataset_1')
    moved = Path(tmp_path, 'dataset_1')
    shutil.move(dataset_dir, moved)
    try:
        loaded = _loaded(config_path)
        loaded.encode(spatial_key_index='spatial_index')
        table = loaded.sdata.tables['phenocoder']
        assert table.shape[1] == 8
        assert np.isfinite(table.X).all()
    finally:
        shutil.move(moved, dataset_dir)


def test_old_model_falls_back_to_the_datasets(config_path, tmp_path):
    """A model directory without the copies (saved before them) reads the datasets."""
    model_dir = config_path.parent
    kept = [Path(model_dir, name) for name in ('stats.csv', 'sample_datasets.csv')]
    for path in kept:
        shutil.move(path, Path(tmp_path, path.name))
    try:
        loaded = _loaded(config_path)
        loaded.encode(spatial_key_index='spatial_index')
        assert 'phenocoder' in loaded.sdata.tables
    finally:
        for path in kept:
            shutil.move(Path(tmp_path, path.name), path)


def test_unseen_sample_is_scaled_to_its_own_image(config_path):
    """A well under a new name is a sample the model never saw. It is scaled to the range of
    its own image (the same range its old name had) and labelled with the only dataset."""
    pheno = example_3d()
    table = pheno.sdata.tables['nuclei_features']
    well = str(table.obs['well'].iloc[0])
    table.obs['well'] = table.obs['well'].astype(str).replace(well, 'unseen')
    pheno.sdata.images['IF_unseen'] = pheno.sdata.images[f'IF_{well}']

    reference = _loaded(config_path)
    reference.encode(spatial_key_index='spatial_index')

    loaded = _loaded(config_path, pheno)
    loaded.encode(spatial_key_index='spatial_index')

    # The reference range went through stats.csv, the unseen one did not: equal up to the
    # last bit pandas' CSV reader may round.
    generator = loaded.patch_generator
    np.testing.assert_allclose(
        generator.sample_percentiles_low['unseen'],
        reference.patch_generator.sample_percentiles_low[well],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        generator.sample_percentiles_high['unseen'],
        reference.patch_generator.sample_percentiles_high[well],
        rtol=1e-12,
    )
    unseen = generator.patches['well'] == 'unseen'
    assert unseen.any()
    assert (generator.patches.loc[unseen, 'dataset'] == 'dataset_1').all()
    assert (loaded.sdata.tables['phenocoder'].obs['well'] == 'unseen').sum() == unseen.sum()


def test_unseen_condition_value_is_encoded_as_zeros(config_path):
    """A z plane the training patches never had gets an all-zero condition, not an error."""
    loaded = _loaded(config_path)
    assert loaded.model_oh_enc.handle_unknown == 'ignore'
    names = list(loaded.model_oh_enc.feature_names_in_)
    known = {name: cats[0] for name, cats in zip(names, loaded.model_oh_enc.categories_)}
    known['z'] = 10_000
    row = loaded.model_oh_enc.transform(pd.DataFrame([known])[names])
    row = row.toarray() if hasattr(row, 'toarray') else row
    z_columns = [
        i
        for i, name in enumerate(loaded.model_oh_enc.get_feature_names_out())
        if name.startswith('z_')
    ]
    assert not row[0, z_columns].any()
