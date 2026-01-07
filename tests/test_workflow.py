import shutil

from spatialdata import SpatialData

from tests.conftest import example_3d


def test_workflow():
    pheno = example_3d()
    assert pheno.sdata is not None
    assert isinstance(pheno.sdata, SpatialData)
    pheno.generate_dataset(
        dataset='dataset_1',
        dir_dataset='tests/data/tmp/phenocoder',
        spatial_key_index='spatial_index',
    )
    pheno.initialize_model(n_latent_dim=32, n_dense_dim=64, conditions=['dataset', 'z'])
    pheno.train(n_epochs=5)
    pheno.encode(spatial_key_index='spatial_index')
    shutil.rmtree('tests/data/tmp/phenocoder')


if __name__ == '__main__':
    test_workflow()
