"""Regression tests for the flip, encode-subset, patch-size and smaller generator fixes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phenocoder.generator import DatasetLoader, PatchGenerator, SequenceGenerator
from tests.conftest import example_3d


def _patch_generator(pheno, patch_size, scale=False, scale_per_sample=True):
    return PatchGenerator(
        sdata=pheno.sdata,
        sample_key='well',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        scale=scale,
        scale_per_sample=scale_per_sample,
        patch_size=patch_size,
    )


def _small_model(pheno, tmp_path, conditions, flip=False):
    """A 40-patch dataset and an (untrained) model on it; batch_size=8 so both splits
    hold complete batches."""
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=(32, 32),
        spatial_key_index='spatial_index',
        n_patches=40,
    )
    pheno.initialize_model(
        n_latent_dim=4,
        n_dense_dim=8,
        conditions=conditions,
        input_shape=(32, 32, 4),
        batch_size=8,
        flip=flip,
    )


# --- fix 5: flip ------------------------------------------------------------------------


@pytest.mark.parametrize('conditions', [[], ['dataset']])
def test_flip_reaches_training_generator_only(tmp_path, conditions):
    """flip=True was dropped for unconditional models, and flipped validation data for
    conditional ones. Now: training flips, validation never does."""
    pheno = example_3d()
    _small_model(pheno, tmp_path, conditions, flip=True)
    assert pheno.data_generator_train.flip is True
    assert pheno.data_generator_val.flip is False


def _write_patches(tmp_path, patches):
    paths = []
    for i, patch in enumerate(patches):
        path = tmp_path / f'p{i}.npy'
        np.save(path, patch)
        paths.append(path)
    return paths


def test_sequence_generator_flips_and_batches_float32(tmp_path):
    """With flip=True some batches come out flipped; batches are float32 (fix 8)."""
    patch = np.arange(16, dtype=np.float16).reshape(1, 4, 4)  # (c, h, w), asymmetric
    paths = _write_patches(tmp_path, [patch])
    plain = np.moveaxis(patch, 0, -1).astype(np.float32)

    generator = SequenceGenerator(
        paths, batch_size=1, dim=(4, 4), n_channels=1, shuffle=False, flip=False
    )
    batch = generator[0]
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0], plain)

    np.random.seed(0)
    generator = SequenceGenerator(
        paths, batch_size=1, dim=(4, 4), n_channels=1, shuffle=False, flip=True
    )
    seen = {generator[0][0].tobytes() for _ in range(50)}
    expected = {
        a.tobytes()
        for a in (
            plain,
            np.fliplr(plain),
            np.flipud(plain),
            np.flipud(np.fliplr(plain)),
        )
    }
    # all four orientations appear, and nothing else
    assert seen == expected


# --- fix 6: encode after generate_dataset(n_patches=...) ----------------------------------


@pytest.mark.parametrize('conditions', [[], ['dataset']])
def test_encode_after_subsampled_dataset_encodes_every_object(tmp_path, conditions):
    pheno = example_3d()
    _small_model(pheno, tmp_path, conditions)
    assert len(pheno.patch_generator.patches) == 40

    full = _patch_generator(pheno, (32, 32))
    full.init_patches()
    assert len(full.patches) > 40

    pheno.encode(spatial_key_index='spatial_index')
    encoded = pheno.sdata.tables['phenocoder']
    assert encoded.n_obs == len(full.patches)
    assert set(encoded.obs_names) == set(full.patches.index)


# --- fix 7: non-square and odd patch sizes ------------------------------------------------


@pytest.mark.parametrize('patch_size', [(24, 40), (33, 33), (31, 20)])
def test_non_square_and_odd_patch_sizes(tmp_path, patch_size):
    """patch_size is (height, width): every patch has that shape, is the crop starting
    half a patch before the centre, and every object whose patch fits is kept."""
    height, width = patch_size
    pheno = example_3d()

    # objects kept: exactly those whose whole patch lies inside their image
    generator = _patch_generator(pheno, patch_size)
    generator.init_patches()
    table = pheno.sdata.tables['nuclei_features']
    coords = pd.DataFrame(
        table.obsm['spatial_index'][:, :2].astype(int),
        columns=['y', 'x'],
        index=table.obs.index,
    )
    coords['well'] = table.obs['well']
    shapes = {
        well: pheno.sdata.images[f'IF_{well}'].shape[-2:]
        for well in coords['well'].unique()
    }
    fits = coords.apply(
        lambda r: (
            height // 2 <= r['y'] < shapes[r['well']][0] - (height - height // 2)
            and width // 2 <= r['x'] < shapes[r['well']][1] - (width - width // 2)
        ),
        axis=1,
    )
    assert set(generator.patches.index) == set(coords.index[fits])

    # patches written: shape and values
    pheno.dir_project = tmp_path
    pheno.generate_dataset(
        dataset='ds',
        patch_size=patch_size,
        spatial_key_index='spatial_index',
        scale=False,
        n_patches=20,
    )
    patches = pd.read_csv(Path(pheno.dir_dataset('ds'), 'patches.csv'), index_col=0)
    for _, row in patches.iterrows():
        patch = np.load(Path(pheno.dir_dataset('ds'), row['file']))
        img = np.asarray(pheno.sdata.images[f'IF_{row["well"]}'])
        y0 = int(row['y']) - height // 2
        x0 = int(row['x']) - width // 2
        crop = img[:, int(row['z']), y0 : y0 + height, x0 : x0 + width]
        assert patch.shape == (4, height, width)
        np.testing.assert_array_equal(patch, crop)


# --- fix 8: smaller ones -------------------------------------------------------------------


def test_mad_is_centred_per_channel():
    """MAD is taken around each channel's median over all of its slices."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 4))
    # two channels at very different intensities, two slices each
    dim = np.stack([base, base + 1])
    bright = np.stack([base + 1000, base + 1001])
    imgs = np.stack([dim, bright])
    stand_in = SimpleNamespace(channels=['c0', 'c1'], scale_percentile=1)
    df = PatchGenerator.__get_image_stats__(stand_in, imgs, 's', 'sample_id')

    expected = [
        np.median(np.abs(sl - np.median(ch))) for ch in (dim, bright) for sl in ch
    ]
    np.testing.assert_allclose(df['mad'].to_numpy(), expected)
    # a channel 1000 apart from the other has the same spread, not a 1000-wide one
    np.testing.assert_allclose(
        df[df['channel'] == 'c0']['mad'].to_numpy(),
        df[df['channel'] == 'c1']['mad'].to_numpy(),
    )


def test_scaling_leaves_percentiles_unchanged():
    """A constant channel (low == high) used to bump percentiles_high in place."""
    pheno = example_3d()
    generator = _patch_generator(pheno, (32, 32), scale=True, scale_per_sample=False)
    generator.init_patches()
    generator.percentiles_low = np.full(4, 5.0)
    generator.percentiles_high = np.full(4, 5.0)

    row = generator.patches.iloc[0]
    img = np.asarray(pheno.sdata.images[f'IF_{row["well"]}'])
    first = generator.extract_patch(img, row['id'])
    second = generator.extract_patch(img, row['id'])

    np.testing.assert_array_equal(generator.percentiles_high, np.full(4, 5.0))
    np.testing.assert_array_equal(first, second)
    y, x, z = int(row['y']), int(row['x']), int(row['z'])
    crop = img[:, z, y - 16 : y + 16, x - 16 : x + 16].astype(np.float32)
    np.testing.assert_array_equal(first, np.clip(crop - 5.0, 0, 1))


def test_initialize_model_reads_datasets_once(tmp_path, monkeypatch):
    calls = []
    load_datasets = DatasetLoader.load_datasets

    def counting(self):
        calls.append(1)
        load_datasets(self)

    monkeypatch.setattr(DatasetLoader, 'load_datasets', counting)
    pheno = example_3d()
    _small_model(pheno, tmp_path, conditions=[])
    assert len(calls) == 1
