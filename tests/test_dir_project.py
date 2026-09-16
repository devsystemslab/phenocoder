"""Tests for the dir_project layout and the dataset bookkeeping that depends on it."""

import shutil
from pathlib import Path

import pytest

import phenocoder as phc
from tests.conftest import example_3d


def test_dir_project_settable_without_generate_dataset(tmp_path):
    """dir_project can be set in the constructor.

    A workflow with no images (e.g. a simulation-derived reference) never calls
    generate_dataset, which used to be the only thing that set the artifact root.
    """
    pheno = phc.Phenocoder(table_key='t', sample_key='s', dir_project=tmp_path)
    assert pheno.dir_project == tmp_path
    assert pheno.dir_reference == Path(tmp_path, 'reference')
    # strings are coerced to Path on assignment
    pheno.dir_project = str(tmp_path)
    assert isinstance(pheno.dir_project, Path)


def test_require_dir_project_raises_when_unset():
    pheno = phc.Phenocoder(table_key='t', sample_key='s')
    assert pheno.dir_project is None
    with pytest.raises(ValueError, match='dir_project must be set'):
        pheno.require_dir_project()


def test_generate_dataset_requires_a_directory():
    """Neither dir_project nor dir_dataset set -> a clear error, not a TypeError."""
    pheno = example_3d()
    with pytest.raises(ValueError, match='Set dir_project'):
        pheno.generate_dataset(dataset='d1', spatial_key_index='spatial_index')


def test_dir_dataset_defaults_under_dir_project(tmp_path):
    pheno = phc.Phenocoder(table_key='t', sample_key='s', dir_project=tmp_path)
    assert pheno.dir_dataset('d1') == Path(tmp_path, 'd1')


def test_dir_dataset_override_is_per_dataset(tmp_path):
    """An explicit dir_dataset overrides for that dataset only, leaving dir_project alone."""
    outside = tmp_path / 'scratch' / 'patches'
    pheno = phc.Phenocoder(
        table_key='t',
        sample_key='s',
        dir_project=tmp_path / 'proj',
        dir_datasets={'d_out': outside},
    )
    assert pheno.dir_dataset('d_out') == outside
    assert pheno.dir_dataset('d_in') == Path(tmp_path, 'proj', 'd_in')
    assert pheno.dir_project == Path(tmp_path, 'proj')


def test_generate_dataset_appends_datasets_without_clobbering():
    """Regression: `self.datasets = self.datasets.append(x)` set datasets to None.

    list.append returns None, so generating a second dataset used to wipe the list.
    """
    pheno = example_3d()
    pheno.dir_project = 'tests/data/tmp'
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
    pheno.dir_project = 'tests/data/tmp'
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

        # dir_project and datasets are recovered from the config's path and contents
        assert loaded.dir_project == Path('tests/data/tmp')
        assert loaded.datasets == ['dataset_1']

        loaded.encode(spatial_key_index='spatial_index')
        assert 'phenocoder' in loaded.sdata.tables
        assert loaded.sdata.tables['phenocoder'].shape[1] == 8
    finally:
        shutil.rmtree('tests/data/tmp', ignore_errors=True)


def test_batch_size_reaches_split_and_generators():
    """Regression: initialize_model called set_train_val_split() with no arguments.

    The split truncation always used the default 64 while the generators batched at the
    configured size, so a non-default batch_size dropped patches to align with the wrong
    number -- and a small dataset was emptied outright.
    """
    pheno = example_3d()
    pheno.dir_project = 'tests/data/tmp'
    try:
        pheno.generate_dataset(
            dataset='dataset_1',
            patch_size=(32, 32),
            spatial_key_index='spatial_index',
            n_patches=40,
        )
        pheno.initialize_model(
            n_latent_dim=8,
            n_dense_dim=16,
            conditions=[],
            input_shape=(32, 32, 4),
            batch_size=8,
        )
        # every split is a whole number of batches at the *configured* size
        sizes = pheno.data_loader.patches.groupby('split').size()
        assert all(n % 8 == 0 for n in sizes)
        assert pheno.data_generator_train.batch_size == 8
        assert len(pheno.data_generator_train) > 0
    finally:
        shutil.rmtree('tests/data/tmp', ignore_errors=True)


def test_load_model_rejects_pre_rename_config(tmp_path):
    """An old `dataset_dirs` key must fail loudly, not be silently ignored.

    Reading it with .get('dir_datasets') would resolve those datasets under dir_project
    instead of their real location.
    """
    import yaml

    config = tmp_path / 'models' / 'm' / 'config.yaml'
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.dump(
            {
                'conditional': False,
                'input_shape': [32, 32, 4],
                'n_latent_dim': 8,
                'n_dense_dim': 16,
                'conv_layers': [8, 16],
                'datasets': ['d1'],
                'dataset_dirs': {'d1': '/scratch/d1'},
            }
        )
    )
    pheno = phc.Phenocoder(table_key='t', sample_key='s')
    pheno.model_config = str(config)
    with pytest.raises(ValueError, match='old "dataset_dirs" key'):
        pheno.load_model()


def test_config_yaml_has_no_absolute_dataset_path():
    """config.yaml stores dataset *names*, so a project directory stays relocatable."""
    pheno = example_3d()
    pheno.dir_project = 'tests/data/tmp'
    try:
        pheno.generate_dataset(
            dataset='dataset_1',
            patch_size=(32, 32),
            spatial_key_index='spatial_index',
            n_patches=64,
        )
        # batch_size=8 so this small dataset still forms complete batches
        pheno.initialize_model(
            n_latent_dim=8,
            n_dense_dim=16,
            conditions=[],
            input_shape=(32, 32, 4),
            batch_size=8,
        )
        assert pheno.model_config['datasets'] == ['dataset_1']
        # only genuine out-of-tree overrides are stored as paths; there are none here
        assert pheno.model_config['dir_datasets'] == {}
        # no absolute path anywhere in the persisted config
        assert not any(
            isinstance(v, str) and v.startswith('/')
            for v in pheno.model_config.values()
        )
    finally:
        shutil.rmtree('tests/data/tmp', ignore_errors=True)
