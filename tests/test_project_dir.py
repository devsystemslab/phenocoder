"""Tests for the project_dir layout and the dataset bookkeeping that depends on it."""

import shutil
from pathlib import Path

import pytest

import phenocoder as phc
from tests.conftest import example_3d


def test_project_dir_settable_without_generate_dataset(tmp_path):
    """project_dir can be set in the constructor.

    A workflow with no images (e.g. a simulation-derived reference) never calls
    generate_dataset, which used to be the only thing that set the artifact root.
    """
    pheno = phc.Phenocoder(table_key='t', sample_key='s', project_dir=tmp_path)
    assert pheno.project_dir == tmp_path
    assert pheno.reference_dir == Path(tmp_path, 'reference')
    # strings are coerced to Path on assignment
    pheno.project_dir = str(tmp_path)
    assert isinstance(pheno.project_dir, Path)


def test_require_project_dir_raises_when_unset():
    pheno = phc.Phenocoder(table_key='t', sample_key='s')
    assert pheno.project_dir is None
    with pytest.raises(ValueError, match='project_dir must be set'):
        pheno.require_project_dir()


def test_generate_dataset_requires_a_directory():
    """Neither project_dir nor dataset_dir set -> a clear error, not a TypeError."""
    pheno = example_3d()
    with pytest.raises(ValueError, match='Set project_dir'):
        pheno.generate_dataset(dataset='d1', spatial_key_index='spatial_index')


def test_dataset_dir_defaults_under_project_dir(tmp_path):
    pheno = phc.Phenocoder(table_key='t', sample_key='s', project_dir=tmp_path)
    assert pheno.dataset_dir('d1') == Path(tmp_path, 'd1')


def test_dataset_dir_override_is_per_dataset(tmp_path):
    """An explicit dataset_dir overrides for that dataset only, leaving project_dir alone."""
    outside = tmp_path / 'scratch' / 'patches'
    pheno = phc.Phenocoder(
        table_key='t',
        sample_key='s',
        project_dir=tmp_path / 'proj',
        dataset_dirs={'d_out': outside},
    )
    assert pheno.dataset_dir('d_out') == outside
    assert pheno.dataset_dir('d_in') == Path(tmp_path, 'proj', 'd_in')
    assert pheno.project_dir == Path(tmp_path, 'proj')


def test_generate_dataset_appends_datasets_without_clobbering():
    """Regression: `self.datasets = self.datasets.append(x)` set datasets to None.

    list.append returns None, so generating a second dataset used to wipe the list.
    """
    pheno = example_3d()
    pheno.project_dir = 'tests/data/tmp'
    try:
        pheno.generate_dataset(
            dataset='ds_a', spatial_key_index='spatial_index', n_patches=4
        )
        assert pheno.datasets == ['ds_a']
        pheno.generate_dataset(
            dataset='ds_b', spatial_key_index='spatial_index', n_patches=4
        )
        assert pheno.datasets == ['ds_a', 'ds_b']
        # regenerating an existing dataset must not duplicate the name
        pheno.generate_dataset(
            dataset='ds_a', spatial_key_index='spatial_index', n_patches=4
        )
        assert pheno.datasets == ['ds_a', 'ds_b']
    finally:
        shutil.rmtree('tests/data/tmp', ignore_errors=True)


def test_load_model_then_encode():
    """Regression: load_model did not restore .datasets, so encode() raised TypeError.

    encode() iterates self.datasets to find patches.csv/stats.csv. load_model only set
    model/model_dir/model_config/oh_encoder, leaving datasets None ->
    "TypeError: 'NoneType' is not iterable".
    """
    pheno = example_3d()
    pheno.project_dir = 'tests/data/tmp'
    try:
        pheno.generate_dataset(
            dataset='dataset_1',
            patch_size=(32, 32),
            spatial_key_index='spatial_index',
        )
        pheno.initialize_model(
            n_latent_dim=8,
            n_dense_dim=16,
            conditions=['dataset', 'z'],
            input_shape=(32, 32, 4),
        )
        pheno.train(n_epochs=1, plot=False)
        config_path = Path(pheno.model_dir, 'config.yaml')

        # a fresh instance that only knows where the config lives
        loaded = example_3d()
        loaded.model_config = str(config_path)
        loaded.load_model()

        # project_dir and datasets are recovered from the config's path and contents
        assert loaded.project_dir == Path('tests/data/tmp')
        assert loaded.datasets == ['dataset_1']

        loaded.encode(spatial_key_index='spatial_index')
        assert 'phenocoder' in loaded.sdata.tables
        assert loaded.sdata.tables['phenocoder'].shape[1] == 8
    finally:
        shutil.rmtree('tests/data/tmp', ignore_errors=True)


def test_config_yaml_has_no_absolute_dataset_path():
    """config.yaml stores dataset *names*, so a project directory stays relocatable."""
    pheno = example_3d()
    pheno.project_dir = 'tests/data/tmp'
    try:
        # NB: no n_patches here. set_train_val_split truncates each split to a whole
        # number of batches, so a dataset smaller than batch_size (64) empties both
        # splits and initialize_model fails inside pandas. Pre-existing, unrelated to
        # project_dir; just don't trip over it here.
        pheno.generate_dataset(
            dataset='dataset_1',
            patch_size=(32, 32),
            spatial_key_index='spatial_index',
        )
        pheno.initialize_model(
            n_latent_dim=8,
            n_dense_dim=16,
            conditions=[],
            input_shape=(32, 32, 4),
        )
        assert pheno.model_config['datasets'] == ['dataset_1']
        assert pheno.model_config['dataset_dirs'] == {}
        assert 'dir_dataset' not in pheno.model_config
    finally:
        shutil.rmtree('tests/data/tmp', ignore_errors=True)
