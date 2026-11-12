"""Tests for patch_transform, skipping statistics with scale=False, and parallel writing."""

import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from phenocoder.generator import (
    DatasetLoader,
    PatchGenerator,
    cut_patch,
    patch_transform_name,
)
from tests.conftest import example_3d

PATCH_SIZE = (32, 32)


# Transforms are module-level so worker processes can unpickle them.
def add_id_channel(patch, obs_row):
    """Append a channel holding the object's instance_id; returns float32."""
    id_channel = np.full(
        (1, *patch.shape[1:]), obs_row['instance_id'], dtype=np.float32
    )
    return np.concatenate([patch.astype(np.float32), id_channel])


def add_zero_channel(patch, obs_row):
    """Same shape as add_id_channel, but a different transform."""
    return np.concatenate(
        [patch.astype(np.float32), np.zeros((1, *patch.shape[1:]), np.float32)]
    )


def _crop(pheno, row):
    """The patch cut straight from the image, for comparison."""
    img = np.asarray(pheno.sdata.images[f'IF_{row["well"]}'])
    half = PATCH_SIZE[0] // 2
    y, x, z = int(row['y']), int(row['x']), int(row['z'])
    return img[:, z, y - half : y + half, x - half : x + half]


def _read_patches(dir_dataset):
    patches = pd.read_csv(Path(dir_dataset, 'patches.csv'), index_col=0)
    # obs names are strings ('0', '1', ...); read_csv parses them as ints
    patches.index = patches.index.astype(str)
    return patches


def test_transform_output_is_written(tmp_path):
    """generate_dataset writes the transform's output: its shape, dtype and values."""
    pheno = example_3d()
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=PATCH_SIZE,
        spatial_key_index='spatial_index',
        scale=False,
        patch_transform=add_id_channel,
        n_patches=20,
    )
    obs = pheno.sdata.tables['nuclei_features'].obs
    patches = _read_patches(pheno.dir_dataset('ds'))
    assert len(patches) == 20
    for obs_name, row in patches.iterrows():
        patch = np.load(Path(pheno.dir_dataset('ds'), row['file']))
        assert patch.shape == (5, *PATCH_SIZE)
        assert patch.dtype == np.float32
        np.testing.assert_array_equal(patch[:4], _crop(pheno, row))
        # the transform saw this object's own obs row
        assert (patch[4] == obs.loc[obs_name, 'instance_id']).all()

    config = yaml.safe_load(Path(pheno.dir_dataset('ds'), 'dataset.yaml').read_text())
    assert config['patch_transform'] == patch_transform_name(add_id_channel)
    assert config['scale'] is False


def test_no_transform_writes_the_plain_crop(tmp_path):
    """Without a transform (the default) a patch is the image crop, dtype unchanged.

    Uses n_patches, so the patch ids are not contiguous and the id lookup is exercised.
    """
    pheno = example_3d()
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=PATCH_SIZE,
        spatial_key_index='spatial_index',
        scale=False,
        n_patches=20,
    )
    patches = _read_patches(pheno.dir_dataset('ds'))
    for _, row in patches.iterrows():
        patch = np.load(Path(pheno.dir_dataset('ds'), row['file']))
        crop = _crop(pheno, row)
        assert patch.dtype == crop.dtype
        np.testing.assert_array_equal(patch, crop)
    config = yaml.safe_load(Path(pheno.dir_dataset('ds'), 'dataset.yaml').read_text())
    assert config['patch_transform'] is None


def test_get_patches_applies_transform():
    """get_patches (used by encode) returns the transform's output, channels last."""
    pheno = example_3d()
    patch_generator = PatchGenerator(
        sdata=pheno.sdata,
        sample_key='well',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        scale=False,
        patch_size=PATCH_SIZE,
        patch_transform=add_id_channel,
    )
    patch_generator.init_patches()
    sample = patch_generator.patches['well'].iloc[0]
    patches, df_patches = patch_generator.get_patches(sample)
    assert patches.shape == (len(df_patches), *PATCH_SIZE, 5)
    obs = pheno.sdata.tables['nuclei_features'].obs
    expected_ids = obs.loc[df_patches.index, 'instance_id'].to_numpy()
    np.testing.assert_array_equal(patches[:, 0, 0, 4], expected_ids)


def test_encode_with_transform(tmp_path):
    """The transform name reaches config.yaml; encode uses the transform and warns on a
    different one, both when reusing the patch generator and when rebuilding it."""
    pheno = example_3d()
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=PATCH_SIZE,
        spatial_key_index='spatial_index',
        scale=False,
        patch_transform=add_id_channel,
    )
    pheno.initialize_model(
        n_latent_dim=4,
        n_dense_dim=8,
        conditions=[],
        input_shape=(*PATCH_SIZE, 5),
        batch_size=8,
    )
    name = patch_transform_name(add_id_channel)
    assert pheno.model_config['patch_transform'] == name
    config = yaml.safe_load(Path(pheno.model_dir, 'config.yaml').read_text())
    assert config['patch_transform'] == name

    n_patches = len(pheno.patch_generator.patches)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pheno.encode(
            scale=False,
            spatial_key_index='spatial_index',
            patch_transform=add_id_channel,
        )
    assert not [w for w in caught if 'patch_transform' in str(w.message)]
    assert pheno.sdata.tables['phenocoder'].shape == (n_patches, 4)

    # rebuilt from scratch (as after load_model), with a different transform
    pheno.patch_generator = None
    with pytest.warns(UserWarning, match='patch_transform'):
        pheno.encode(
            scale=False,
            spatial_key_index='spatial_index',
            patch_transform=add_zero_channel,
        )
    assert pheno.patch_generator.patch_transform is add_zero_channel


def test_scale_false_skips_statistics(tmp_path, monkeypatch):
    """With scale=False no statistics are computed or written, training works without
    stats.csv, and encode(scale=True) says clearly that stats.csv is missing."""

    def fail(*args, **kwargs):
        raise AssertionError('image statistics computed with scale=False')

    monkeypatch.setattr(PatchGenerator, 'generate_image_stats', fail)
    pheno = example_3d()
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=PATCH_SIZE,
        spatial_key_index='spatial_index',
        scale=False,
    )
    assert not Path(pheno.dir_dataset('ds'), 'stats.csv').exists()

    pheno.initialize_model(
        n_latent_dim=4,
        n_dense_dim=8,
        conditions=[],
        input_shape=(*PATCH_SIZE, 4),
        batch_size=8,
    )
    assert pheno.data_loader.stats is None
    pheno.train(n_epochs=1, plot=False)

    pheno.patch_generator = None
    with pytest.raises(ValueError, match='stats.csv'):
        pheno.encode(scale=True, spatial_key_index='spatial_index')


def test_scale_true_still_writes_statistics(tmp_path):
    """The default (scale=True) keeps writing stats.csv."""
    pheno = example_3d()
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=PATCH_SIZE,
        spatial_key_index='spatial_index',
        n_patches=8,
    )
    assert Path(pheno.dir_dataset('ds'), 'stats.csv').exists()


@pytest.mark.parametrize('transform', [None, add_id_channel])
def test_parallel_writes_identical_files(tmp_path, transform):
    """n_workers=2 writes byte-identical patches, patches.csv and stats.csv to n_workers=1.

    scale=True so the per-sample scaling ranges sent to the workers are exercised too.
    """
    pheno = example_3d()
    pheno.dir_project = tmp_path
    dirs = {}
    for n_workers in (1, 2):
        dirs[n_workers] = tmp_path / f'workers_{n_workers}'
        pheno.generate_dataset(
            dataset='ds',
            dir_dataset=dirs[n_workers],
            patch_size=PATCH_SIZE,
            spatial_key_index='spatial_index',
            patch_transform=transform,
            n_workers=n_workers,
        )
    files = {n: sorted(p.name for p in d.iterdir()) for n, d in dirs.items()}
    assert files[1] == files[2]
    assert any(name.endswith('.npy') for name in files[1])
    for name in files[1]:
        assert (dirs[1] / name).read_bytes() == (dirs[2] / name).read_bytes(), name


def test_parallel_rejects_unpicklable_transform(tmp_path):
    pheno = example_3d()
    pheno.dir_project = tmp_path
    with pytest.raises(ValueError, match='module-level function'):
        pheno.generate_dataset(
            dataset='ds',
            patch_size=PATCH_SIZE,
            spatial_key_index='spatial_index',
            scale=False,
            patch_transform=lambda patch, obs_row: patch,
            n_workers=2,
        )


class Identity:
    def __call__(self, patch, obs_row):
        return patch


def test_patch_transform_name():
    assert patch_transform_name(None) is None
    assert patch_transform_name(add_id_channel) == (
        f'{add_id_channel.__module__}.add_id_channel'
    )
    partial = functools.partial(add_id_channel, obs_row=None)
    assert patch_transform_name(partial) == (
        f'functools.partial({add_id_channel.__module__}.add_id_channel, obs_row=None)'
    )
    # a callable object is named by its class
    assert patch_transform_name(Identity()) == f'{Identity.__module__}.Identity'


def test_transform_must_return_three_dimensions():
    img = np.zeros((2, 8, 8))
    with pytest.raises(ValueError, match=r'\(c, h, w\)'):
        cut_patch(img, 4, 4, 0, (4, 4), patch_transform=lambda patch, row: patch[0])


def test_datasets_with_different_transforms_are_rejected():
    loader = DatasetLoader(
        datasets=['a', 'b', 'c'],
        dir_datasets={'a': '.', 'b': '.', 'c': '.'},
        sample_key='s',
    )
    loader.dataset_configs = {
        'a': {'patch_transform': 'm.f'},
        'b': {'patch_transform': 'm.f'},
    }
    # 'c' has no dataset.yaml: counts as no transform, which differs from 'm.f'
    with pytest.raises(ValueError, match='different patch transforms'):
        loader.get_patch_transform()
    loader.datasets = ['a', 'b']
    assert loader.get_patch_transform() == 'm.f'
