import shutil
from pathlib import Path

import pandas as pd

from phenocoder import generator
from tests.conftest import example_3d


def test_dataset_generator():
    pheno = example_3d()
    data_generator = generator.PatchGenerator(
        sdata=pheno.sdata,
        sample_key='well',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        scale=True,
    )
    data_generator.generate_dataset(dataset='test_dataset', dir_output='tests/data/tmp')
    shutil.rmtree(data_generator.dir_output)


def test_dataset_generator_n_samples():
    """generate_dataset(n_samples=1) restricts processing to a single sample.

    ``n_samples`` selects which samples are processed and written to disk, so
    exactly one sample is selected and only its patch ``.npy`` files are written.
    """
    pheno = example_3d()
    n_available = (
        pheno.sdata.tables['nuclei_features'].obs['well'].nunique()
    )
    assert n_available > 1  # the test data must have several samples to subset
    data_generator = generator.PatchGenerator(
        sdata=pheno.sdata,
        sample_key='well',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        scale=True,
    )
    data_generator.generate_dataset(
        dataset='test_dataset', dir_output='tests/data/tmp', n_samples=1
    )
    # only one sample was selected for processing
    assert len(data_generator.samples) == 1
    selected = str(data_generator.samples[0])
    # and only that sample's patches were written to disk as .npy files
    npy_files = list(Path(data_generator.dir_dataset).glob('*.npy'))
    assert len(npy_files) > 0
    assert all(f.name.startswith(f'{selected}_') for f in npy_files)
    shutil.rmtree(data_generator.dir_output)


def test_dataset_generator_n_patches():
    """generate_dataset(n_patches=N) restricts output to N patches."""
    n_patches = 10
    pheno = example_3d()
    data_generator = generator.PatchGenerator(
        sdata=pheno.sdata,
        sample_key='well',
        image_key='IF',
        spatial_key='spatial_index',
        table_key='nuclei_features',
        scale=True,
    )
    data_generator.generate_dataset(
        dataset='test_dataset', dir_output='tests/data/tmp', n_patches=n_patches
    )
    patches = pd.read_csv(Path(data_generator.dir_dataset, 'patches.csv'))
    assert patches.shape[0] == n_patches
    shutil.rmtree(data_generator.dir_output)


def test_phenocoder_generate_dataset_sampling_args():
    """Phenocoder.generate_dataset forwards n_samples/n_patches without error.

    Guards the regression where these kwargs were wrongly passed to the
    PatchGenerator constructor (which does not accept them).
    """
    n_patches = 8
    pheno = example_3d()
    pheno.generate_dataset(
        dataset='test_dataset',
        dir_dataset='tests/data/tmp',
        spatial_key_index='spatial_index',
        n_patches=n_patches,
    )
    patches = pd.read_csv(Path(pheno.patch_generator.dir_dataset, 'patches.csv'))
    assert patches.shape[0] == n_patches
    shutil.rmtree('tests/data/tmp')


if __name__ == '__main__':
    test_dataset_generator()
    test_dataset_generator_n_samples()
    test_dataset_generator_n_patches()
    test_phenocoder_generate_dataset_sampling_args()
