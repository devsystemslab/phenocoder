import shutil

from tests.conftest import example_3d


def test_train():
    pheno = example_3d()
    pheno.generate_dataset(
        dataset='test_dataset',
        dir_dataset='tests/data/tmp',
        spatial_key_index='spatial_index',
    )
    pheno.initialize_model(n_latent_dim=32, n_dense_dim=64, conditions=['dataset', 'z'])
    pheno.train(n_epochs=3)
    shutil.rmtree('tests/data/tmp')


if __name__ == '__main__':
    test_train()
